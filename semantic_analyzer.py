"""
Semantic analysis helpers for role-aware KV-cache eviction.

This module provides two static signals computed from the prefill prompt:
1. Role tags for ChatML-style conversations.
2. Token-local information density scores.
"""

import re
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

        # Detect chat template format: ChatML (Qwen) vs Llama-style
        self.chat_format = "chatml"
        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        if self.im_start_id is None or self.im_end_id is None:
            # Try Llama-style tokens
            start_header = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
            end_header = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
            eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if start_header is not None and eot is not None:
                self.chat_format = "llama"
                self.im_start_id = start_header
                self.im_end_id = eot
                self.llama_end_header_id = end_header
            else:
                self.im_start_id = -1
                self.im_end_id = -1

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
        all_known_ids = [self.im_start_id, self.im_end_id, *self.tokenizer.all_special_ids, *role_token_ids, 0]
        max_known_token_id = max(tid for tid in all_known_ids if tid is not None and tid >= 0)
        feature_size = max(
            int(getattr(self.tokenizer, "vocab_size", 0)),
            len(self.tokenizer),
            max_known_token_id + 1,
        )

        self.is_digit_token = torch.zeros(feature_size, dtype=torch.bool)
        self.is_punct_token = torch.zeros(feature_size, dtype=torch.bool)
        self.is_code_token = torch.zeros(feature_size, dtype=torch.bool)
        self.is_upper_token = torch.zeros(feature_size, dtype=torch.bool)
        self.is_month_token = torch.zeros(feature_size, dtype=torch.bool)
        self.is_fact_unit_token = torch.zeros(feature_size, dtype=torch.bool)
        self.is_entityish_token = torch.zeros(feature_size, dtype=torch.bool)
        self.is_chat_boundary_token = torch.zeros(feature_size, dtype=torch.bool)
        self.is_chat_template_token = torch.zeros(feature_size, dtype=torch.bool)

        for token_id in self.tokenizer.all_special_ids:
            if 0 <= token_id < feature_size:
                self.is_chat_boundary_token[token_id] = True
                self.is_chat_template_token[token_id] = True
        for token_id in [self.im_start_id, self.im_end_id]:
            if 0 <= token_id < feature_size:
                self.is_chat_boundary_token[token_id] = True
                self.is_chat_template_token[token_id] = True
        for token_ids in self.role_token_map.values():
            for token_id in token_ids:
                if 0 <= token_id < feature_size:
                    self.is_chat_template_token[token_id] = True

        code_chars = set("{}[]()=><;:#_/\\@$%^&*|~`")
        month_names = {
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        }
        fact_units = {
            "am",
            "pm",
            "utc",
            "hour",
            "hours",
            "hr",
            "hrs",
            "minute",
            "minutes",
            "min",
            "mins",
            "degree",
            "degrees",
            "celsius",
            "fahrenheit",
        }

        # Batch-decode all tokens at once instead of one-by-one loop over vocab
        all_token_ids = list(range(feature_size))
        all_token_strs = self.tokenizer.batch_decode(
            [[tid] for tid in all_token_ids],
            skip_special_tokens=False,
        )
        for token_id, token_str in enumerate(all_token_strs):
            if not token_str:
                continue

            if token_str.strip() == "" and (
                len(token_str) > 1 or any(char in token_str for char in "\n\r\t")
            ):
                self.is_chat_template_token[token_id] = True

            if any(char.isdigit() for char in token_str):
                self.is_digit_token[token_id] = True

            stripped = token_str.strip()
            if stripped and all((not char.isalnum()) and char != " " for char in stripped):
                self.is_punct_token[token_id] = True

            if any(char in code_chars for char in token_str):
                self.is_code_token[token_id] = True

            if any(char.isupper() for char in token_str):
                self.is_upper_token[token_id] = True

            if self._looks_entityish(token_str):
                self.is_entityish_token[token_id] = True

            normalized = re.sub(r"[^a-z]", "", token_str.lower())
            if normalized in month_names:
                self.is_month_token[token_id] = True
            if normalized in fact_units:
                self.is_fact_unit_token[token_id] = True

    @staticmethod
    def _looks_entityish(token_str: str) -> bool:
        """Heuristic for tokens that look like names, labels, or anchored identifiers."""
        stripped = token_str.strip()
        if not stripped:
            return False

        if re.search(r"[A-Z][a-z]+(?:[-'][A-Za-z0-9]+)?", stripped):
            return True
        if re.search(r"[A-Z]{2,}", stripped):
            return True
        if re.search(r"[A-Za-z]+[-:][A-Za-z0-9]+", stripped):
            return True
        if re.search(r"[A-Za-z]+-\d+", stripped):
            return True
        return False

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
        """Detect the role token sequence after a header-start token.

        ChatML: <|im_start|>system\\n ...
        Llama:  <|start_header_id|>system<|end_header_id|>\\n ...
        """
        if self.chat_format == "llama":
            # Scan tokens between start_header and end_header
            end = pos
            while end < len(ids) and ids[end] != getattr(self, "llama_end_header_id", -1):
                end += 1
            span = ids[pos:end]
            for role_name, role_ids in self.role_token_map.items():
                if span == role_ids:
                    return role_name
            return "unknown"

        # ChatML: role tokens immediately follow <|im_start|>
        for role_name, role_ids in self.role_token_map.items():
            if pos + len(role_ids) <= len(ids) and ids[pos : pos + len(role_ids)] == role_ids:
                return role_name
        return "unknown"

    def _find_im_end(self, ids: list[int], start: int) -> Optional[int]:
        """Find the next end-of-turn token index (<|im_end|> or <|eot_id|>)."""
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

    def compute_query_relevance(
        self,
        input_ids: torch.Tensor,
        query_text: str,
        window_size: int = 32,
    ) -> torch.Tensor:
        """Estimate token-level relevance to the final question text."""
        seq_len = input_ids.shape[0]
        relevance = torch.zeros(seq_len, dtype=torch.float32)
        query_text = query_text.strip()
        if not query_text:
            return relevance

        query_token_ids = set(self.tokenizer.encode(query_text, add_special_tokens=False))
        if not query_token_ids:
            return relevance

        ids = input_ids.cpu().tolist()
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2)
            window = ids[start:end]
            if not window:
                continue

            overlap = sum(1 for token_id in window if token_id in query_token_ids)
            relevance[i] = overlap / len(window)

        return relevance

    def compute_factual_bonus(
        self,
        input_ids: torch.Tensor,
        window_size: int = 32,
    ) -> torch.Tensor:
        """Estimate how likely each position belongs to a factual span."""
        ids = input_ids.cpu()
        seq_len = len(ids)
        bonus = torch.zeros(seq_len, dtype=torch.float32)

        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2)
            window = ids[start:end]
            w_len = len(window)
            if w_len == 0:
                continue

            digit_ratio = self._feature_ratio(self.is_digit_token, window)
            upper_ratio = self._feature_ratio(self.is_upper_token, window)
            month_ratio = self._feature_ratio(self.is_month_token, window)
            fact_unit_ratio = self._feature_ratio(self.is_fact_unit_token, window)
            entity_ratio = self._feature_ratio(self.is_entityish_token, window)
            bonus[i] = min(
                1.0,
                0.25 * digit_ratio
                + 0.12 * upper_ratio
                + 0.28 * month_ratio
                + 0.28 * fact_unit_ratio
                + 0.45 * entity_ratio,
            )

        return bonus

    def compute_authority_bonus(
        self,
        input_ids: torch.Tensor,
        query_text: str,
        window_size: int = 32,
    ) -> torch.Tensor:
        """Boost spans that align with source-of-truth cues from the question."""
        seq_len = input_ids.shape[0]
        bonus = torch.zeros(seq_len, dtype=torch.float32)
        query_lower = query_text.lower()
        if not query_lower:
            return bonus

        authority_terms = [
            "current",
            "authoritative",
            "approved",
            "source-of-truth",
            "source of truth",
            "exact",
            "actual",
            "real",
            "official",
        ]
        active_terms = [term for term in authority_terms if term in query_lower]
        if not active_terms:
            return bonus

        authority_token_ids: set[int] = set()
        for term in active_terms:
            authority_token_ids.update(self.tokenizer.encode(term, add_special_tokens=False))
        authority_token_ids.discard(None)
        if not authority_token_ids:
            return bonus

        ids = input_ids.cpu().tolist()
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2)
            window = ids[start:end]
            if not window:
                continue

            overlap = sum(1 for token_id in window if token_id in authority_token_ids)
            bonus[i] = overlap / len(window)

        return bonus

    def compute_latest_question_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Mark the question/instruction tail inside the latest user segment."""
        ids = input_ids.cpu().tolist()
        seq_len = len(ids)
        mask = torch.zeros(seq_len, dtype=torch.bool)

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

        latest_user_segment = None
        for start, role_name, end in reversed(segments):
            if role_name == "user":
                latest_user_segment = (start, end)
                break
        if latest_user_segment is None:
            return mask

        seg_start, seg_end = latest_user_segment
        segment_ids = ids[seg_start : seg_end + 1]
        markers = [
            "Now answer this question:",
            "Question:",
            "Q:",
        ]
        marker_start = None
        for marker in markers:
            marker_ids = self.tokenizer.encode(marker, add_special_tokens=False)
            if not marker_ids or len(marker_ids) > len(segment_ids):
                continue
            for offset in range(0, len(segment_ids) - len(marker_ids) + 1):
                if segment_ids[offset : offset + len(marker_ids)] == marker_ids:
                    marker_start = seg_start + offset
                    break
            if marker_start is not None:
                break

        if marker_start is not None:
            mask[marker_start : seg_end + 1] = True
            return mask

        fallback_len = min(48, seg_end - seg_start + 1)
        if fallback_len > 0:
            mask[seg_end - fallback_len + 1 : seg_end + 1] = True
        return mask

    def compute_question_like_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Mark spans that look like questions/prompts instead of factual statements."""
        ids = input_ids.cpu().tolist()
        seq_len = len(ids)
        mask = torch.zeros(seq_len, dtype=torch.bool)

        cue_phrases = [
            "what",
            "when",
            "where",
            "which",
            "who",
            "how",
            "can you",
            "could you",
            "would you",
            "tell me",
            "specific aspects",
        ]
        cue_token_spans = [
            self.tokenizer.encode(phrase, add_special_tokens=False)
            for phrase in cue_phrases
        ]
        cue_token_spans = [span for span in cue_token_spans if span]

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

        for seg_start, _role_name, seg_end in segments:
            segment_ids = ids[seg_start : seg_end + 1]
            for cue_ids in cue_token_spans:
                if len(cue_ids) > len(segment_ids):
                    continue
                for offset in range(0, len(segment_ids) - len(cue_ids) + 1):
                    if segment_ids[offset : offset + len(cue_ids)] == cue_ids:
                        global_start = seg_start + offset
                        global_end = min(seg_end + 1, global_start + max(12, len(cue_ids) + 12))
                        mask[global_start:global_end] = True
            for idx in range(seg_start, seg_end + 1):
                token_text = self.tokenizer.decode([ids[idx]])
                if "?" in token_text:
                    local_start = max(seg_start, idx - 10)
                    local_end = min(seg_end + 1, idx + 3)
                    mask[local_start:local_end] = True

        return mask

    def compute_chat_template_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return a mask for ChatML/template-formatting tokens."""
        token_ids = input_ids.to(dtype=torch.long).cpu()
        valid = (token_ids >= 0) & (token_ids < len(self.is_chat_template_token))
        mask = torch.zeros_like(token_ids, dtype=torch.bool)
        if valid.any():
            mask[valid] = self.is_chat_template_token[token_ids[valid]]
        return mask

    def compute_chat_boundary_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return a narrow mask for the boundary tokens required by ChatML structure."""
        token_ids = input_ids.to(dtype=torch.long).cpu()
        valid = (token_ids >= 0) & (token_ids < len(self.is_chat_boundary_token))
        mask = torch.zeros_like(token_ids, dtype=torch.bool)
        if valid.any():
            mask[valid] = self.is_chat_boundary_token[token_ids[valid]]
        return mask

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
