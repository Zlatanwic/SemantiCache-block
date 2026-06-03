import torch

from attention_tracker import AttentionTracker
from eviction_policies import OPSieveKVLitePolicy
from semantic_analyzer import SemanticAnalyzer


class FakeTokenizer:
    def __init__(self):
        self.id_to_token = {
            0: "<pad>",
            1: "<|im_start|>",
            2: "<|im_end|>",
            3: "system",
            4: "user",
            5: "assistant",
            10: "Alpha",
            11: " clue",
            12: ".",
            13: " filler",
            14: " Beta",
            15: " context",
            16: " noise",
            17: "?",
            18: " answer",
            19: " 37",
            20: " UTC",
            21: " irrelevant",
        }
        self.token_to_id = {token: token_id for token_id, token in self.id_to_token.items()}
        self.all_special_ids = [0, 1, 2]
        self.vocab_size = max(self.id_to_token) + 1

    def __len__(self):
        return self.vocab_size

    def convert_tokens_to_ids(self, token):
        return self.token_to_id.get(token, -1)

    def encode(self, text, add_special_tokens=False, return_tensors=None):
        if text in self.token_to_id:
            ids = [self.token_to_id[text]]
        else:
            ids = [token_id for token_id, token in self.id_to_token.items() if token.strip() and token.strip() in text]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids, skip_special_tokens=False):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        text = "".join(self.id_to_token.get(int(token_id), "") for token_id in ids)
        if skip_special_tokens:
            text = text.replace("<|im_start|>", "").replace("<|im_end|>", "").replace("<pad>", "")
        return text

    def batch_decode(self, batch_ids, skip_special_tokens=False):
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_ids]


def make_policy(input_ids):
    tokenizer = FakeTokenizer()
    analyzer = SemanticAnalyzer(tokenizer)
    tracker = AttentionTracker(num_layers=1, num_kv_heads=1)
    tracker.per_head_scores = torch.zeros(1, 1, len(input_ids), dtype=torch.float32)

    policy = OPSieveKVLitePolicy(
        tracker=tracker,
        analyzer=analyzer,
        pin_system=False,
        pin_latest_user=False,
        recent_window_size=0,
        generated_retention_window=0,
        max_segment_tokens=4,
        min_segment_tokens=2,
    )
    policy.setup_semantic_signals(input_ids, latest_query_text="37 UTC")
    return policy


def test_op_sievekv_lite_keeps_high_value_segment():
    input_ids = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], dtype=torch.long)
    policy = make_policy(input_ids)

    policy.query_relevance = torch.zeros(len(input_ids), dtype=torch.float32)
    policy.factual_bonus = torch.zeros(len(input_ids), dtype=torch.float32)
    policy.authority_bonus = torch.zeros(len(input_ids), dtype=torch.float32)
    policy.info_density = torch.zeros(len(input_ids), dtype=torch.float32)
    policy.query_relevance[8:11] = 1.0
    policy.factual_bonus[8:11] = 1.0

    scores = policy.compute_eviction_scores(len(input_ids))
    keep = policy.select_keep_indices(scores, budget=5)

    assert {8, 9, 10}.issubset(set(keep.tolist()))
    assert keep.numel() == 5


def test_kvtip_marks_confident_wrong_decisions():
    input_ids = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17], dtype=torch.long)
    policy = make_policy(input_ids)

    policy.query_relevance = torch.zeros(len(input_ids), dtype=torch.float32)
    policy.factual_bonus = torch.zeros(len(input_ids), dtype=torch.float32)
    scores = policy.compute_eviction_scores(len(input_ids))
    assert scores.numel() == len(input_ids)

    oracle = torch.zeros(len(input_ids), dtype=torch.float32)
    oracle[4:6] = 1.0
    stats = policy.compute_kvtip_stats(oracle)

    assert stats.soft_or.shape == oracle.shape
    assert torch.all(stats.soft_or >= 0)
    assert torch.all(stats.soft_or <= 1)
    assert (stats.quadrant == 3).any()
