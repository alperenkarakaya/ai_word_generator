"""
Text generation with the trained Transformer.

Public API
----------
TransformerEngine(checkpoint_path, tokenizer_path)
  .generate(prompt, max_new_tokens, ...)           -> str  (generated portion only)
  .generate_until_sentence_end(prompt, ...)        -> str  (stops at TR001 / TR003 / TR004)
  .generate_paragraph(prompt, max_sentences, ...)  -> str  (stops at TR017 or max_sentences)

Generation algorithm
--------------------
1. Encode the prompt with SentencePiece → token-id list.
2. Run one full forward pass on the entire prompt → logits + KV-cache.
3. Sample the next token from logits[:, -1, :] using top-k + top-p + repetition penalty.
4. Feed that single token back in with the cached key/value tensors (fast: one
   transformer pass instead of reprocessing the full context).
5. Repeat until a stop token is sampled or max_new_tokens is reached.
6. When the running context length would exceed block_size, slide the window:
   keep the last (block_size − 1) tokens, reset the cache, and continue.
"""

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from text_utils import (
    SENTENCE_END_TOKENS,
    PARAGRAPH_BREAK_TOKENS,
    decode,
    encode,
    load_tokenizer,
    restore_punctuation_from_tokens,
    tr_token_ids,
)
from transformer.model import GPT, GPTConfig


@dataclass
class SamplingConfig:
    max_new_tokens: int = 80
    temperature: float = 0.85      # > 1 → more random, < 1 → more focused
    top_k: int = 40                # keep only top-k candidates
    top_p: float = 0.92            # nucleus: drop the tail below cumulative prob p
    repetition_penalty: float = 1.15  # penalise tokens already in context


class TransformerEngine:
    """
    Loads a trained GPT checkpoint and exposes sentence / paragraph generation.

    Parameters
    ----------
    checkpoint_path : str
        Path to the .pt file saved by transformer/train.py.
    tokenizer_path : str
        Path to the SentencePiece .model file built by tokenizer/build_tokenizer.py.
    device : str, optional
        'cuda' or 'cpu'. Auto-detected if omitted.
    """

    def __init__(self, checkpoint_path: str, tokenizer_path: str,
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sp = load_tokenizer(tokenizer_path)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device)

        # Sanity check: tokenizer must match the one used during training.
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

        # Pre-compute the integer ids for every special TR token so the
        # generation loop can check stop conditions without string comparisons.
        self._tr_ids = tr_token_ids()
        self._sentence_end_ids = {
            self._tr_ids[t] for t in SENTENCE_END_TOKENS if t in self._tr_ids
        }
        self._paragraph_end_ids = {
            self._tr_ids[t] for t in PARAGRAPH_BREAK_TOKENS if t in self._tr_ids
        }

        print(
            f"TransformerEngine ready — "
            f"{self.model.num_parameters() / 1e6:.1f}M params, "
            f"device={self.device}, block_size={self.cfg.block_size}"
        )

    # ------------------------------------------------------------------ #
    #  Sampling helpers                                                    #
    # ------------------------------------------------------------------ #

    def _sample_one(self, logits: torch.Tensor, history: torch.Tensor,
                    sc: SamplingConfig) -> int:
        """
        Draw one token from `logits` using the sampling config.

        Steps (applied in order):
          1. Repetition penalty — reduce the logit of every token that already
             appears in `history` to discourage repetitive output.
          2. Temperature — divide all logits by T before softmax. T < 1 sharpens
             the distribution (more deterministic); T > 1 flattens it (more creative).
          3. Top-k — zero out every logit below the k-th highest.
          4. Top-p (nucleus) — sort by probability; keep the smallest set whose
             cumulative probability ≥ p, discard the rest.
          5. Softmax + multinomial sample.
        """
        logits = logits.clone().float()  # fp32 for numerical stability during sampling

        # 1. Repetition penalty
        if sc.repetition_penalty != 1.0 and history.numel() > 0:
            unique_ids = torch.unique(history)
            scores = logits[unique_ids]
            # Positive logits → divide; negative logits → multiply.
            logits[unique_ids] = torch.where(
                scores > 0,
                scores / sc.repetition_penalty,
                scores * sc.repetition_penalty,
            )

        # 2. Temperature
        if sc.temperature > 0:
            logits = logits / sc.temperature

        # 3. Top-k
        if 0 < sc.top_k < logits.size(-1):
            kth_value = torch.topk(logits, sc.top_k).values[-1]
            logits[logits < kth_value] = float("-inf")

        # 4. Top-p (nucleus)
        if 0.0 < sc.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens whose cumulative prob exceeds p (shifted right by 1 so
            # we always keep at least the top-1 token).
            remove_mask = cumulative_probs > sc.top_p
            remove_mask[1:] = remove_mask[:-1].clone()
            remove_mask[0] = False

            sorted_logits[remove_mask] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(0, sorted_indices, sorted_logits)

        # 5. Sample
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def _encode_prompt(self, text: str) -> list[int]:
        """Text → token-id list, trimmed to block_size if too long."""
        ids = encode(text)
        if not ids:
            bos = self.sp.bos_id()
            ids = [bos] if bos >= 0 else [2]  # fallback to id=2 (BOS set in build_tokenizer)
        return ids[-self.cfg.block_size:]  # keep right-most tokens if too long

    # ------------------------------------------------------------------ #
    #  Core generation loop                                                #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _generate_ids(self, prompt_ids: list[int], sc: SamplingConfig,
                      stop_ids: Optional[set] = None) -> list[int]:
        """
        Autoregressive generation with KV-cache and sliding-window fallback.

        How the KV-cache works
        ----------------------
        A transformer's attention layers compute Key and Value matrices for every
        token in the context.  On the first forward pass (the full prompt), these
        matrices are returned as `kv_caches`.  On every subsequent step, we only
        feed the single newly sampled token; the model appends its K and V to the
        cache and skips recomputing everything before it.  This makes each step
        O(context_length) in memory but O(1) in compute relative to the sequence.

        Sliding window
        --------------
        The positional embedding only has `block_size` slots (0 … block_size−1).
        Once the context length reaches block_size, we cannot extend further without
        the position index going out of range.  We then drop the oldest tokens,
        keeping the last (block_size − 1) ones, reset the cache, and continue.
        The model loses the very earliest context but retains recent coherence.

        Parameters
        ----------
        prompt_ids : list of int
            Encoded prompt token ids.
        sc : SamplingConfig
            Sampling hyperparameters.
        stop_ids : set of int, optional
            Generation stops immediately when any of these token ids is sampled.

        Returns
        -------
        list of int  (newly generated ids only, not including the prompt)
        """
        # all_tokens tracks the full token history (prompt + generated).
        # It is used to build the history tensor for repetition penalty.
        all_tokens = list(prompt_ids)
        new_ids: list[int] = []

        # ── Step 1: process the full prompt in one forward pass ──────────
        inp = torch.tensor([all_tokens], dtype=torch.long, device=self.device)
        logits, _, kv = self.model(inp, kv_caches=None, pos_offset=0)
        # n_cached = number of token positions currently stored in the KV-cache.
        # Positions in the cache are 0 … n_cached−1.
        n_cached = len(all_tokens)

        # ── Step 2: sample and generate tokens one at a time ─────────────
        for step in range(sc.max_new_tokens):
            # Sample next token from logits at the last position.
            history = torch.tensor(all_tokens, dtype=torch.long, device=self.device)
            next_id = self._sample_one(logits[0, -1, :], history, sc)

            all_tokens.append(next_id)
            new_ids.append(next_id)

            # Check stop condition BEFORE running another forward pass.
            if stop_ids and next_id in stop_ids:
                break
            if step == sc.max_new_tokens - 1:
                break  # quota exhausted

            # ── Feed next_id back through the model to get logits for the
            #    token after it. ─────────────────────────────────────────
            if n_cached >= self.cfg.block_size:
                # Sliding window: the KV-cache is full.
                # Keep the most recent (block_size − 1) tokens + next_id at the end
                # so the full window of block_size tokens is re-processed from pos 0.
                window = all_tokens[-self.cfg.block_size:]
                inp = torch.tensor([window], dtype=torch.long, device=self.device)
                logits, _, kv = self.model(inp, kv_caches=None, pos_offset=0)
                n_cached = len(window)  # = block_size (KV-cache reset)
            else:
                # Normal incremental step: feed only next_id.
                # Its position is n_cached (the slot immediately after the cache).
                inp = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                logits, _, kv = self.model(inp, kv_caches=kv, pos_offset=n_cached)
                n_cached += 1  # cache now includes next_id

        return new_ids

    # ------------------------------------------------------------------ #
    #  Public generation API                                               #
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str, max_new_tokens: int = 80, **kwargs) -> str:
        """
        Generate `max_new_tokens` tokens after `prompt`. No stop condition.

        Returns the generated text only (prompt not repeated).
        """
        sc = SamplingConfig(max_new_tokens=max_new_tokens, **kwargs)
        ids = self._generate_ids(self._encode_prompt(prompt), sc, stop_ids=None)
        return decode(ids)

    def generate_until_sentence_end(self, prompt: str,
                                    max_new_tokens: int = 100, **kwargs) -> str:
        """
        Generate tokens until a sentence-ending punctuation token is produced:
          TR001 = '.'   TR003 = '!'   TR004 = '?'

        This mirrors the contract of the old NgramPredictor.predict_until_sentence_end.
        """
        sc = SamplingConfig(max_new_tokens=max_new_tokens, **kwargs)
        ids = self._generate_ids(
            self._encode_prompt(prompt), sc, stop_ids=self._sentence_end_ids
        )
        return decode(ids)

    def generate_paragraph(self, prompt: str, max_sentences: int = 5,
                            max_new_tokens: int = 300, **kwargs) -> str:
        """
        Generate text until a paragraph-break token (TR017 = newline) is produced
        OR `max_sentences` sentence-enders have been emitted — whichever comes first.

        Used by /predict_paragraph (Phase 3).
        """
        sc = SamplingConfig(max_new_tokens=max_new_tokens,
                            temperature=kwargs.pop("temperature", 0.75),
                            top_p=kwargs.pop("top_p", 0.90),
                            **kwargs)
        prompt_ids = self._encode_prompt(prompt)
        all_stop = self._sentence_end_ids | self._paragraph_end_ids

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

            if next_id in self._paragraph_end_ids:
                break
            if next_id in self._sentence_end_ids:
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

        return decode(new_ids)
