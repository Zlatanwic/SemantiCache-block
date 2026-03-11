"""
Semantic analysis helpers for role-aware KV-cache eviction.

This module provides two static signals computed from the prefill prompt:
1. Role tags for ChatML-style conversations.
2. Token-local information density scores.
"""

from enum import IntEnum
from typing import Optional

import torch
from transformers import PreTrainedTokenizer


class RoleTag(IntEnum):
    """Larger values correspond to higher protection priority."""

    FILLER = 0
    ASSISTANT = 1
    CONTEXT = 2
    USER_HISTORY = 3
    USER_LATEST = 4
    SYSTEM = 5


class SemanticAnalyzer:
    """Compute role tags and information-density signals from a prompt."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        self.role_token_map: dict[str, list[int]] = {}
        for role_name in ["system", "user", "assistant"]:
            self.role_token_map[role_name] = tokenizer.encode(
                role_name,
                add_special_tokens=False,
            )

        self._build_token_features()

    def _build_token_features(self) -> None:
        """Precompute token-type features used by information-density scoring."""
        role_token_ids = [token_id for ids in self.role_token_map.values() for token_id in ids]
        max_known_token_id = max(
            [self.im_start_id, self.im_end_id, *self.tokenizer.all_special_ids, *role_token_ids, 0]
        )
        feature_size = max(
            int(getattr(self.tokenizer, "vocab_size", 0)),
            len(self.tokenizer),
            max_known_token_id + 1,
        )

        self.is_digit_token = torch.zeros(feature_size, dtype=torch.bool)
        self.is_punct_token = torch.zeros(feature_size, dtype=torch.bool)
        self.is_code_token = torch.zeros(feature_size, dtype=torch.bool)

        code_chars = set("{}[]()=><;:#_/\\@$%^&*|~`")
        for token_id in range(feature_size):
            try:
                token_str = self.tokenizer.decode([token_id])
            except Exception:
                continue

            if any(char.isdigit() for char in token_str):
                self.is_digit_token[token_id] = True

            stripped = token_str.strip()
            if stripped and all((not char.isalnum()) and char != " " for char in stripped):
                self.is_punct_token[token_id] = True

            if any(char in code_chars for char in token_str):
                self.is_code_token[token_id] = True

    @staticmethod
    def _feature_ratio(feature_table: torch.Tensor, token_ids: torch.Tensor) -> float:
        """Return the fraction of tokens whose ids map to True in a feature table."""
        if token_ids.numel() == 0:
            return 0.0

        token_ids = token_ids.to(dtype=torch.long)
        valid = (token_ids >= 0) & (token_ids < len(feature_table))
        if not valid.any():
            return 0.0

        matched = feature_table[token_ids[valid]].float().sum().item()
        return matched / token_ids.numel()

    def compute_role_tags(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Assign a role tag to each token in a ChatML-formatted prompt."""
        ids = input_ids.cpu().tolist()
        seq_len = len(ids)
        tags = torch.full((seq_len,), RoleTag.FILLER, dtype=torch.long)

        segments: list[tuple[int, str, int]] = []
        i = 0
        while i < seq_len:
            if ids[i] == self.im_start_id:
                seg_start = i
                role_name = self._detect_role(ids, i + 1)
                end_pos = self._find_im_end(ids, i + 1)
                if end_pos is None:
                    end_pos = seq_len - 1
                segments.append((seg_start, role_name, end_pos))
                i = end_pos + 1
            else:
                i += 1

        last_user_idx = -1
        for idx, (_, role, _) in enumerate(segments):
            if role == "user":
                last_user_idx = idx

        for idx, (start, role, end) in enumerate(segments):
            if role == "system":
                tags[start : end + 1] = RoleTag.SYSTEM
            elif role == "user":
                if idx == last_user_idx:
                    tags[start : end + 1] = RoleTag.USER_LATEST
                else:
                    tags[start : end + 1] = RoleTag.USER_HISTORY
            elif role == "assistant":
                tags[start : end + 1] = RoleTag.ASSISTANT
            else:
                tags[start : end + 1] = RoleTag.CONTEXT

        return tags

    def _detect_role(self, ids: list[int], pos: int) -> str:
        """Detect the role token sequence immediately after `<|im_start|>`."""
        for role_name, role_ids in self.role_token_map.items():
            if pos + len(role_ids) <= len(ids) and ids[pos : pos + len(role_ids)] == role_ids:
                return role_name
        return "unknown"

    def _find_im_end(self, ids: list[int], start: int) -> Optional[int]:
        """Find the next `<|im_end|>` token index."""
        for idx in range(start, len(ids)):
            if ids[idx] == self.im_end_id:
                return idx
        return None

    def compute_info_density(
        self,
        input_ids: torch.Tensor,
        window_size: int = 32,
    ) -> torch.Tensor:
        """
        Estimate a local information-density score for each token.

        Scores are based on token-type ratios within a sliding window:
        - digits increase density
        - code-like symbols increase density
        - punctuation decreases density
        - lexical diversity increases density
        """
        ids = input_ids.cpu()
        seq_len = len(ids)
        density = torch.zeros(seq_len, dtype=torch.float32)

        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2)
            window = ids[start:end]
            w_len = len(window)
            if w_len == 0:
                continue

            digit_ratio = self._feature_ratio(self.is_digit_token, window)
            code_ratio = self._feature_ratio(self.is_code_token, window)
            punct_ratio = self._feature_ratio(self.is_punct_token, window)
            unique_ratio = len(set(window.tolist())) / w_len

            score = (
                0.3 * digit_ratio
                + 0.2 * code_ratio
                - 0.2 * punct_ratio
                + 0.3 * unique_ratio
            )
            density[i] = max(0.0, min(1.0, score))

        return density

    def get_pinned_mask(
        self,
        role_tags: torch.Tensor,
        pin_system: bool,
        pin_latest_user: bool,
        latest_user_tail_tokens: int | None = None,
    ) -> torch.Tensor:
        """Return a mask of tokens that should never be evicted."""
        pinned = torch.zeros_like(role_tags, dtype=torch.bool)

        if pin_system:
            pinned |= role_tags == RoleTag.SYSTEM
        if pin_latest_user:
            latest_user_mask = role_tags == RoleTag.USER_LATEST
            if latest_user_tail_tokens is None or latest_user_tail_tokens <= 0:
                pinned |= latest_user_mask
            else:
                latest_user_indices = torch.nonzero(latest_user_mask, as_tuple=False).flatten()
                if latest_user_indices.numel() <= latest_user_tail_tokens:
                    pinned |= latest_user_mask
                else:
                    pinned[latest_user_indices[-latest_user_tail_tokens:]] = True

        return pinned
