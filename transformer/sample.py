"""
Text generation with the trained Transformer.

Public API
----------
TransformerEngine(checkpoint_path, tokenizer_path)
  .generate(prompt, max_new_tokens, ...)           -> str
  .generate_until_sentence_end(prompt, ...)        -> str  (stops when decoded text ends with . ! ?)
  .generate_paragraph(prompt, max_sentences, ...)  -> str  (stops at newline or max_sentences)

Pure BPE mode: sentence-end detection is done on the decoded string
(SENTENCE_END_CHARS = '.', '!', '?') rather than on special token ids.
"""

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from text_utils import (
    PARAGRAPH_BREAK_CHAR,
    SENTENCE_END_CHARS,
    decode,
    encode,
    load_tokenizer,
    text_ends_sentence,
)
from transformer.model import GPT, GPTConfig


@dataclass
class SamplingConfig:
    max_new_tokens: int = 80
    temperature: float = 0.85
    top_k: int = 40
    top_p: float = 0.92
    repetition_penalty: float = 1.15


class TransformerEngine:
    """Loads a trained GPT checkpoint and exposes sentence / paragraph generation."""

    def __init__(self, checkpoint_path: str, tokenizer_path: str,
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sp = load_tokenizer(tokenizer_path)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device)

        if ckpt["config"]["vocab_size"] != self.sp.get_piece_size():
            raise ValueError(
                f"Checkpoint vocab_size ({ckpt['config']['vocab_size']}) does not "
                f"match tokenizer ({self.sp.get_piece_size()}). "
                f"Rebuild the tokenizer or use the matching checkpoint."
            )

        self.cfg = GPTConfig(**ckpt["config"])
        self.model = GPT(self.cfg).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        print(
            f"TransformerEngine ready — "
            f"{self.model.num_parameters() / 1e6:.1f}M params, "
            f"device={self.device}, block_size={self.cfg.block_size}"
        )

    # ------------------------------------------------------------------ #
    #  Sampling helpers                                                   #
    # ------------------------------------------------------------------ #

    def _sample_one(self, logits: torch.Tensor, history: torch.Tensor,
                    sc: SamplingConfig) -> int:
        logits = logits.clone().float()

        if sc.repetition_penalty != 1.0 and history.numel() > 0:
            unique_ids = torch.unique(history)
            scores = logits[unique_ids]
            logits[unique_ids] = torch.where(
                scores > 0,
                scores / sc.repetition_penalty,
                scores * sc.repetition_penalty,
            )

        if sc.temperature > 0:
            logits = logits / sc.temperature

        if 0 < sc.top_k < logits.size(-1):
            kth_value = torch.topk(logits, sc.top_k).values[-1]
            logits[logits < kth_value] = float("-inf")

        if 0.0 < sc.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cumulative_probs > sc.top_p
            remove_mask[1:] = remove_mask[:-1].clone()
            remove_mask[0] = False
            sorted_logits[remove_mask] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(0, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def _encode_prompt(self, text: str) -> list[int]:
        ids = encode(text)
        if not ids:
            bos = self.sp.bos_id()
            ids = [bos] if bos >= 0 else [2]
        return ids[-self.cfg.block_size:]

    # ------------------------------------------------------------------ #
    #  Core generation loop                                               #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _generate_loop(self, prompt_ids: list[int], sc: SamplingConfig,
                       stop_on_sentence: bool = False,
                       stop_on_paragraph: bool = False,
                       max_sentences: Optional[int] = None) -> list[int]:
        """
        Autoregressive generation with KV-cache and sliding-window fallback.

        Stop conditions (checked by decoding the generated tail each step):
          - stop_on_sentence=True  → stop once decoded tail ends with . ! or ?
          - stop_on_paragraph=True → stop at newline character in decoded tail
          - max_sentences set      → stop after that many sentence-enders
        """
        all_tokens = list(prompt_ids)
        new_ids: list[int] = []

        inp = torch.tensor([all_tokens], dtype=torch.long, device=self.device)
        logits, _, kv = self.model(inp, kv_caches=None, pos_offset=0)
        n_cached = len(all_tokens)
        sentences_ended = 0

        for step in range(sc.max_new_tokens):
            history = torch.tensor(all_tokens, dtype=torch.long, device=self.device)
            next_id = self._sample_one(logits[0, -1, :], history, sc)
            all_tokens.append(next_id)
            new_ids.append(next_id)

            # Decode the running generated tail to check stop conditions on text.
            tail = decode(new_ids)

            if stop_on_paragraph and PARAGRAPH_BREAK_CHAR in tail:
                break
            if stop_on_sentence and text_ends_sentence(tail):
                if max_sentences is None:
                    break
                sentences_ended += 1
                if sentences_ended >= max_sentences:
                    break

            if step == sc.max_new_tokens - 1:
                break

            if n_cached >= self.cfg.block_size:
                window = all_tokens[-self.cfg.block_size:]
                inp = torch.tensor([window], dtype=torch.long, device=self.device)
                logits, _, kv = self.model(inp, kv_caches=None, pos_offset=0)
                n_cached = len(window)
            else:
                inp = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                logits, _, kv = self.model(inp, kv_caches=kv, pos_offset=n_cached)
                n_cached += 1

        return new_ids

    # ------------------------------------------------------------------ #
    #  Public generation API                                              #
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str, max_new_tokens: int = 80, **kwargs) -> str:
        sc = SamplingConfig(max_new_tokens=max_new_tokens, **kwargs)
        ids = self._generate_loop(self._encode_prompt(prompt), sc)
        return decode(ids)

    def generate_until_sentence_end(self, prompt: str,
                                    max_new_tokens: int = 100, **kwargs) -> str:
        sc = SamplingConfig(max_new_tokens=max_new_tokens, **kwargs)
        ids = self._generate_loop(
            self._encode_prompt(prompt), sc, stop_on_sentence=True
        )
        return decode(ids)

    def generate_paragraph(self, prompt: str, max_sentences: int = 5,
                            max_new_tokens: int = 300, **kwargs) -> str:
        sc = SamplingConfig(
            max_new_tokens=max_new_tokens,
            temperature=kwargs.pop("temperature", 0.75),
            top_p=kwargs.pop("top_p", 0.90),
            **kwargs,
        )
        ids = self._generate_loop(
            self._encode_prompt(prompt), sc,
            stop_on_sentence=True,
            stop_on_paragraph=True,
            max_sentences=max_sentences,
        )
        return decode(ids)
