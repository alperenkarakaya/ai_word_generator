"""
N-gram Language Model with Stupid Backoff (orders 1..max_n).
Primary text generation engine for the AI Text Generator.

Stupid Backoff: at generation time, try the highest-order n-gram that has
a matching context; fall back one order at a time until unigram.
"""
import math
import os
import pickle
import random
import re
from collections import Counter, defaultdict, deque

from text_utils import (
    full_clean,
    is_punctuation_token,
    is_sentence_ending,
    replace_punctuation_with_tokens,
    restore_punctuation_from_tokens,
)

# ── hierarchical-sampler tuning (sweep these for the thesis) ──────────────────
GAMMA       = 1.3   # Stage-1 order-preference exponent: score_n ∝ n**GAMMA.
                    #   higher  → stronger pull toward higher-order n-grams.
KAPPA       = 3.0   # Stage-1 confidence saturation: conf_n = 1 - exp(-C_n/KAPPA).
                    #   down-weights high orders backed by only 1-2 observations.
TOPK        = 5     # Stage-2 candidate truncation (keep top-K by count).
REP_WINDOW  = 8     # repetition guard: how many recent words to remember.
REP_PENALTY = 3.0   # repetition guard: divide a candidate's count by this if it
                    #   appeared in the recent window (soft, survives temperature).


class NgramPredictor:
    """N-gram language model with Stupid Backoff from order max_n down to unigram."""

    def __init__(self, load_from_pickle=None, max_n=7):
        self.max_n = max_n
        self.word_counts    = Counter()
        self.unigram_counts = Counter()
        self.bigram_counts  = defaultdict(Counter)   # order 2 — also drives probability panel
        self.trigram_counts = defaultdict(Counter)   # order 3 — also drives probability panel
        self.ngram_counts   = {}   # {n: defaultdict(Counter)}  for n in 4..max_n
        self.total_words    = 0

        if load_from_pickle:
            if os.path.exists(load_from_pickle):
                self.load_from_pickle(load_from_pickle)
            else:
                print(f"Warning: {load_from_pickle} not found, starting empty.")

    # ── helpers ────────────────────────────────────────────────────────────────

    def _prepare_for_lookup(self, text: str) -> list:
        """Convert user text into a token list matching the N-gram model's format."""
        text = full_clean(text, lowercase=True)
        text = replace_punctuation_with_tokens(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.split() if text else []

    # ── hierarchical sampler (generation only — NOT ghost) ──────────────────────

    @staticmethod
    def _tempered(scores: dict, temperature: float) -> dict:
        """
        Unified tempered-softmax-over-scores operator shared by both sampling
        stages:  q_i = x_i**(1/T) / Σ_j x_j**(1/T).

          T → 0⁺  → collapses to argmax (the old deterministic behaviour)
          T = 1   → proportional to the raw scores
          T → ∞   → uniform

        `scores` maps key → positive number (count, or order-preference weight).
        Returns key → probability (sums to 1).
        """
        T = max(temperature, 1e-3)
        powered = {k: (max(v, 1e-12) ** (1.0 / T)) for k, v in scores.items()}
        tot = sum(powered.values()) or 1.0
        return {k: p / tot for k, p in powered.items()}

    def _cands_at_level(self, context: list, n: int):
        """Return the candidate Counter for order n given context, or None.

        All tables store Counters, so callers can use .most_common() directly.
        Level 1 (unigram) is context-independent and always available."""
        if n == 1:
            return self.unigram_counts or None
        if n == 2:
            return self.bigram_counts.get(context[-1]) if context else None
        if n == 3:
            if len(context) < 2:
                return None
            return self.trigram_counts.get((context[-2], context[-1]))
        # n >= 4
        if n not in self.ngram_counts or len(context) < n - 1:
            return None
        return self.ngram_counts[n].get(tuple(context[-(n - 1):]))

    def _level_scores(self, context: list) -> dict:
        """
        Stage-1 raw scores per available order:

            score_n = (n ** GAMMA) * (1 - exp(-C_n / KAPPA))

        where C_n is the total evidence (sum of counts) at that context's
        n-gram entry.  The order term favours higher n; the confidence term
        suppresses high orders that were only seen once or twice (which would
        otherwise force verbatim, memorised continuations).
        Returns {n: score} for every order that has candidates.
        """
        scores = {}
        for n in range(1, self.max_n + 1):
            cands = self._cands_at_level(context, n)
            if not cands:
                continue
            c_n  = sum(cands.values())
            conf = 1.0 - math.exp(-c_n / KAPPA)
            scores[n] = (n ** GAMMA) * conf
        return scores

    def _sample_token(self, context: list, temperature: float = 1.0, recent=None,
                      rng=None):
        """
        Hierarchical probabilistic next-token sampler used by sentence,
        paragraph, and ghost-text generation.

        Stage 1 — sample which n-gram order to use from the tempered level
                  distribution τ(score_n ; T).
        Stage 2 — within that order, keep the top-K candidates by count, apply a
                  repetition penalty, and sample from τ(count_i ; T).

        `rng` defaults to the shared `random` module (fresh randomness every
        call — used by sentence/paragraph generation). Ghost text passes a
        seeded `random.Random` instance instead so the suggestion is
        reproducible for a given (text, temperature) pair and doesn't flicker
        to a different random draw on every keystroke.
        """
        rng = rng or random
        recent = recent or ()
        level_scores = self._level_scores(context)
        if not level_scores:
            return None

        # Stage 1 — choose an order.
        level_probs = self._tempered(level_scores, temperature)
        chosen_n = rng.choices(
            list(level_probs), weights=list(level_probs.values()), k=1
        )[0]

        # Stage 2 — choose a word within that order.
        cands = self._cands_at_level(context, chosen_n)
        word_scores = {}
        for w, c in cands.most_common(TOPK):
            penalty = REP_PENALTY if w in recent else 1.0
            word_scores[w] = c / penalty
        word_probs = self._tempered(word_scores, temperature)
        return rng.choices(
            list(word_probs), weights=list(word_probs.values()), k=1
        )[0]

    def _selection_distribution(self, context: list, temperature: float = 1.0, top_n: int = 8):
        """
        The TRUE next-word distribution the hierarchical sampler draws from:

            P(w) = Σ_n  P(level=n) · P_tempered(w | top-K of order n)

        This composes Stage-1 (which order) with Stage-2 (which word), both at
        the given temperature — i.e. exactly what `_sample_token` samples from,
        minus the runtime repetition penalty (which depends on generation
        history and so has no meaning while the user is merely typing).

        Reuses `_level_scores`, `_tempered` and `TOPK` so the panel and the
        generator can never drift apart. Returns a list of
        {word, probability(%), sources:[{level, pct}]} sorted by probability.
        """
        level_scores = self._level_scores(context)
        if not level_scores:
            return []
        level_probs = self._tempered(level_scores, temperature)   # Stage 1

        combined      = defaultdict(float)          # word -> total prob mass
        contributions = defaultdict(dict)           # word -> {order: prob mass}
        for n, p_level in level_probs.items():
            cands = self._cands_at_level(context, n)
            if not cands:
                continue
            word_scores = {w: c for w, c in cands.most_common(TOPK)}
            word_probs  = self._tempered(word_scores, temperature)   # Stage 2
            for w, pw in word_probs.items():
                mass = p_level * pw
                combined[w]         += mass
                contributions[w][n]  = contributions[w].get(n, 0.0) + mass

        ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        out = []
        for w, p in ranked:
            srcs = sorted(contributions[w].items(), key=lambda kv: kv[1], reverse=True)
            out.append({
                "word": restore_punctuation_from_tokens(w) if is_punctuation_token(w) else w,
                "probability": round(p * 100, 1),
                "sources": [{"level": n, "pct": round(m / p * 100)} for n, m in srcs] if p > 0 else [],
            })
        return out

    @staticmethod
    def _tokens_to_text(tokens: list) -> str:
        """
        Convert a list of raw model tokens to display text.

        Punctuation tokens (TR001–TR017) are attached directly to the preceding
        word without a leading space; all other tokens are space-separated.

        This avoids the ordering-dependency bug in restore_punctuation_from_tokens
        where one TR token immediately followed by another could be missed.

        Examples
        --------
        ["cat", "sat", "TR001"]            → "cat sat."
        ["TR011", "TR002", "known", "as"]  → "), known as"
        ["sat", "on", "the", "mat", "TR001", "he"] → "sat on the mat. he"
        """
        from text_utils import TOKEN_TO_PUNCTUATION
        words = []
        for tok in tokens:
            if is_punctuation_token(tok):
                punct = TOKEN_TO_PUNCTUATION.get(tok, tok)
                if words:
                    words[-1] = words[-1] + punct   # attach to previous word
                else:
                    words.append(punct)              # punctuation at very start
            else:
                words.append(tok)
        return " ".join(words)

    # ── training ───────────────────────────────────────────────────────────────

    def train_from_file(self, filepath):
        """Build n-gram tables for orders 1..max_n from a whitespace-tokenised file."""
        print(f"Training {self.max_n}-gram model from: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        tokens = text.split()
        self.total_words = len(tokens)
        print(f"   Total tokens: {self.total_words:,}")

        self.word_counts.update(tokens)
        self.unigram_counts.update(tokens)

        print("   Building bigram (2-gram) index...")
        for i in range(len(tokens) - 1):
            self.bigram_counts[tokens[i]][tokens[i + 1]] += 1

        print("   Building trigram (3-gram) index...")
        for i in range(len(tokens) - 2):
            self.trigram_counts[(tokens[i], tokens[i + 1])][tokens[i + 2]] += 1

        self.ngram_counts = {}
        for n in range(4, self.max_n + 1):
            print(f"   Building {n}-gram index...")
            table = defaultdict(Counter)
            for i in range(len(tokens) - n + 1):
                ctx = tuple(tokens[i: i + n - 1])
                table[ctx][tokens[i + n - 1]] += 1
            self.ngram_counts[n] = table

        print(f"Training complete. {len(self.unigram_counts):,} unique tokens.")
        for n, tbl in [(2, self.bigram_counts), (3, self.trigram_counts)]:
            print(f"   {n}-gram contexts: {len(tbl):,}")
        for n in range(4, self.max_n + 1):
            print(f"   {n}-gram contexts: {len(self.ngram_counts[n]):,}")

    # ── loading ────────────────────────────────────────────────────────────────

    def load_from_pickle(self, filepath):
        print(f"Loading N-gram model: {filepath}")
        with open(filepath, "rb") as f:
            loaded = pickle.load(f)

        if isinstance(loaded, dict):
            self.word_counts    = Counter(loaded.get("word_counts", {}))
            self.unigram_counts = Counter(loaded.get("unigram_counts", {}))
            self.total_words    = loaded.get("total_words", 0)
            self.max_n          = loaded.get("max_n", 3)

            self.bigram_counts = defaultdict(Counter)
            for k, v in loaded.get("bigram_counts", {}).items():
                self.bigram_counts[k] = Counter(v)

            self.trigram_counts = defaultdict(Counter)
            for k, v in loaded.get("trigram_counts", {}).items():
                self.trigram_counts[k] = Counter(v)

            self.ngram_counts = {}
            for n, raw in loaded.get("ngram_counts", {}).items():
                table = defaultdict(Counter)
                for k, v in raw.items():
                    table[k] = Counter(v)
                self.ngram_counts[int(n)] = table
        elif hasattr(loaded, "__dict__"):
            self.__dict__.update(loaded.__dict__)

        print(f"N-gram model loaded. Vocab: {len(self.word_counts):,}, max_n: {self.max_n}")

    # ── generation ─────────────────────────────────────────────────────────────

    def predict(self, text):
        """Ghost-text word completion (not directly used by generation endpoints)."""
        if not text or text.endswith(" "):
            return ""
        tokens = self._prepare_for_lookup(text)
        if not tokens:
            return ""
        current = tokens[-1]
        if is_punctuation_token(current):
            return ""
        matches = [w for w in self.word_counts if w.startswith(current)]
        if not matches:
            return ""
        best     = max(matches, key=lambda w: self.word_counts[w])
        suffix   = best[len(current):]
        cands    = self.bigram_counts.get(best)
        next_tok = ""
        if cands:
            next_tok = " " + random.choices(list(cands), weights=list(cands.values()), k=1)[0]
        return restore_punctuation_from_tokens(suffix + next_tok)

    def predict_until_sentence_end(self, text, max_words=50, temperature: float = 1.0):
        """
        Generate tokens using Stupid Backoff until a sentence-ending punctuation
        is produced or max_words is reached.
        Returns a string suffix to append to the current editor text.
        """
        tokens = self._prepare_for_lookup(text)
        if not tokens:
            return ""

        context   = list(tokens)
        generated = []

        trailing_space = text.endswith(" ")
        if not trailing_space:
            last = context[-1]
            if not is_punctuation_token(last):
                matches = [w for w in self.word_counts if w.startswith(last)]
                if matches:
                    best   = max(matches, key=lambda w: self.word_counts[w])
                    suffix = best[len(last):]
                    if suffix:
                        generated.append(suffix)
                    context[-1] = best

        # If cursor was at end of a complete word (no suffix was appended),
        # the result must start with a space so words don't run together.
        need_leading_space = not trailing_space and not generated

        recent: deque = deque(maxlen=REP_WINDOW)   # words for the repetition penalty
        produced_bigrams: set = set()              # anti-loop guard

        for _ in range(max_words):
            nw = self._sample_token(context, temperature, recent)
            if nw is None:
                break

            # A sentence can't consist of just a lone punctuation mark — resample
            # a few times before giving up on this sentence entirely.
            resample_attempts = 0
            while not generated and is_sentence_ending(nw) and resample_attempts < 5:
                nw = self._sample_token(context, temperature, recent)
                if nw is None:
                    break
                resample_attempts += 1
            if nw is None or (not generated and is_sentence_ending(nw)):
                break

            # Anti-loop: if this (prev, nw) bigram was already emitted, resample a
            # few times before giving up.  Soft and bounded so it can't spin.
            prev = context[-1] if context else None
            attempts = 0
            while (prev, nw) in produced_bigrams and attempts < 3:
                nw = self._sample_token(context, temperature, recent)
                if nw is None:
                    break
                attempts += 1
            if nw is None:
                break

            generated.append(nw)
            context.append(nw)
            if prev is not None:
                produced_bigrams.add((prev, nw))
            if not is_punctuation_token(nw):
                recent.append(nw)
            if is_sentence_ending(nw):
                break

        result = " ".join(generated)
        if need_leading_space and result:
            result = " " + result
        return restore_punctuation_from_tokens(result)

    def predict_paragraph(self, text, max_sentences=5, max_words_per_sentence=50, temperature: float = 1.0):
        """
        Chain predict_until_sentence_end to produce a multi-sentence paragraph.
        Each call uses the full accumulated text as context so later sentences
        are conditioned on earlier ones.
        """
        accumulated = text
        completions = []
        for _ in range(max_sentences):
            comp = self.predict_until_sentence_end(accumulated, max_words=max_words_per_sentence, temperature=temperature)
            if not comp.strip():
                break
            completions.append(comp)
            accumulated = accumulated + comp
        return "".join(completions)

    # ── ghost text (inline word suggestion) ───────────────────────────────────

    def get_ghost_text(self, text: str, max_ghost_words: int = 7,
                       temperature: float = 1.0) -> str:
        """
        Returns the full ghost-layer string showing up to max_ghost_words words ahead.

        Uses the same hierarchical sampler as sentence/paragraph generation
        (_sample_token), so the inline suggestion reacts to `temperature` just
        like the rest of the UI.

        Algorithm
        ---------
        Step 1 — partial-word resolution:
            If the cursor is inside an incomplete word (e.g. "ca"), find the
            most-frequent word that starts with that prefix ("cat") and make it
            the first ghost token.  The running context is updated so that step 2
            predicts *after* "cat", not after the raw prefix "ca".  This step
            stays deterministic — it's literal word completion, not prediction.

        Step 2 — tempered chain generation:
            Call _sample_token(context, temperature, recent) up to max_ghost_words
            times, with an anti-loop guard so the ghost doesn't repeat the user's
            typed bigrams.  Each new token is appended to `context` so word n+1 is
            conditioned on words 1..n (not just the original user text).  Stop
            early when a sentence-ending token (TR001 / TR003 / TR004) is produced
            — the period is *included* so the user sees where the sentence ends.

        Step 3 — assemble:
            Join ghost tokens with spaces, restore TR tokens to real punctuation,
            then attach to the correct prefix of the user text.

        The returned string is the FULL ghost-layer content (user text + suggestion).
        The JS slices it at len(user_text) to colour the suggestion part gray.
        """
        tokens = self._prepare_for_lookup(text)
        if not text.strip() or not tokens:
            return ""

        trailing_space = text.endswith(" ")
        last = tokens[-1]

        # Never suggest anything after a punctuation token (e.g. a period)
        if not trailing_space and is_punctuation_token(last):
            return ""

        ghost_tokens = []            # raw model tokens that form the gray suggestion
        context      = list(tokens)  # running context — grows as we generate

        # ── Step 1: resolve partial word ─────────────────────────────────────
        if not trailing_space and last not in self.word_counts:
            matches = [w for w in self.word_counts if w.startswith(last)]
            if not matches:
                return ""
            best = max(matches, key=lambda w: self.word_counts[w])
            ghost_tokens.append(best)
            context[-1] = best      # "ca" → "cat" so step-2 context is correct

        # ── Step 2: tempered chain generation ────────────────────────────────
        # Same hierarchical sampler as sentence/paragraph generation, so the
        # ghost suggestion actually reacts to `temperature`. Anti-loop guard
        # still applies so the ghost doesn't repeat the user's typed bigrams.
        #
        # Seeded RNG (keyed on text+temperature) instead of the shared `random`
        # module: the same editor state must keep showing the same suggestion
        # while the user reads it, or Tab can accept a different draw than
        # what was on screen. Changing the temperature slider *does* reroll it,
        # since that changes the seed.
        rng = random.Random((text, round(temperature, 2)))
        remaining = max_ghost_words - len(ghost_tokens)

        seen_bigrams: set = set()
        for i in range(len(tokens) - 1):
            seen_bigrams.add((tokens[i], tokens[i + 1]))

        recent: deque = deque(maxlen=REP_WINDOW)

        for _ in range(remaining):
            tok = self._sample_token(context, temperature, recent, rng=rng)
            if tok is None:
                break

            prev = context[-1] if context else None
            attempts = 0
            while prev is not None and (prev, tok) in seen_bigrams and attempts < 3:
                tok = self._sample_token(context, temperature, recent, rng=rng)
                if tok is None:
                    break
                attempts += 1
            if tok is None or (prev is not None and (prev, tok) in seen_bigrams):
                break

            ghost_tokens.append(tok)
            context.append(tok)
            if not is_punctuation_token(tok):
                recent.append(tok)
            if len(context) >= 2:
                seen_bigrams.add((context[-2], context[-1]))
            if is_sentence_ending(tok):
                break

        if not ghost_tokens:
            return ""

        # ── Step 3: build ghost_full string ──────────────────────────────────
        display        = self._tokens_to_text(ghost_tokens)
        first_is_punct = is_punctuation_token(ghost_tokens[0])

        if trailing_space:
            # "the cat |…"  → "the cat sat on the mat."
            return text + display
        elif last in self.word_counts:
            # "the cat|"    → "the cat sat on the mat."
            # "the said|"   → "the said, he …"  (no space before comma)
            sep = "" if first_is_punct else " "
            return text + sep + display
        else:
            # "the ca|"     → "the cat sat on the mat."
            last_space = text.rfind(" ")
            prefix = text[: last_space + 1] if last_space >= 0 else ""
            return prefix + display

    # ── probability panel (UI) ─────────────────────────────────────────────────

    def get_probabilities(self, text, use_tokens=True, temperature: float = 1.0):
        """Return unigram / bigram / trigram probability tables for the UI panels.

        Each per-level card shows the RAW empirical distribution (count +
        count-normalised %) for the current context — temperature-independent,
        the honest statistical view.  The separate `levels` field is the
        temperature-aware Stage-1 distribution P(level=n) that the *generator*
        would sample from, which drives the Level Selection panel."""
        tokens = self._prepare_for_lookup(text) if text.strip() else []

        total = sum(self.unigram_counts.values()) or 1
        unigram_probs = [
            {
                "word": restore_punctuation_from_tokens(w) if is_punctuation_token(w) else w,
                "probability": round(c / total * 100, 1),
                "count": c,
            }
            for w, c in self.unigram_counts.most_common(10)
        ]

        if not tokens:
            return {
                "unigram": unigram_probs, "bigram": [], "trigram": [],
                "current_word": "", "context": "", "levels": [], "selection": [],
            }

        last = tokens[-1]

        bigram_probs = []
        cands = self.bigram_counts.get(last, {})
        if cands:
            tot = sum(cands.values())
            for w, c in Counter(cands).most_common(5):
                bigram_probs.append({
                    "word": restore_punctuation_from_tokens(w) if is_punctuation_token(w) else w,
                    "probability": round(c / tot * 100, 1),
                    "count": c,
                })

        trigram_probs = []
        ctx_display = ""
        if len(tokens) >= 2:
            key   = (tokens[-2], tokens[-1])
            cands = self.trigram_counts.get(key, {})
            if cands:
                tot = sum(cands.values())
                for w, c in Counter(cands).most_common(5):
                    trigram_probs.append({
                        "word": restore_punctuation_from_tokens(w) if is_punctuation_token(w) else w,
                        "probability": round(c / tot * 100, 1),
                        "count": c,
                    })
            w1 = restore_punctuation_from_tokens(tokens[-2]) if is_punctuation_token(tokens[-2]) else tokens[-2]
            w2 = restore_punctuation_from_tokens(tokens[-1]) if is_punctuation_token(tokens[-1]) else tokens[-1]
            ctx_display = f"{w1} {w2}"

        display_last = restore_punctuation_from_tokens(last) if is_punctuation_token(last) else last

        # 4-gram through 7-gram panels
        _order_names = {4: "fourgram", 5: "fivegram", 6: "sixgram", 7: "sevengram"}
        _ctx_names   = {4: "context4", 5: "context5", 6: "context6", 7: "context7"}
        higher = {}
        for n in range(4, self.max_n + 1):
            name     = _order_names.get(n)
            ctx_name = _ctx_names.get(n)
            if name is None:
                continue
            probs   = []
            ctx_str = ""
            if len(tokens) >= n - 1 and n in self.ngram_counts:
                key   = tuple(tokens[-(n - 1):])
                cands = self.ngram_counts[n].get(key, {})
                if cands:
                    tot = sum(cands.values())
                    for w, c in Counter(cands).most_common(5):
                        probs.append({
                            "word": restore_punctuation_from_tokens(w) if is_punctuation_token(w) else w,
                            "probability": round(c / tot * 100, 1),
                            "count": c,
                        })
                ctx_tokens = tokens[-(n - 1):]
                ctx_str = " ".join(
                    restore_punctuation_from_tokens(t) if is_punctuation_token(t) else t
                    for t in ctx_tokens
                )
            higher[name]     = probs
            higher[ctx_name] = ctx_str

        # ── Stage-1 level distribution P(level=n) at this temperature ─────────
        levels = []
        level_scores = self._level_scores(tokens)
        if level_scores:
            level_probs = self._tempered(level_scores, temperature)
            for n in sorted(level_probs, reverse=True):
                cands = self._cands_at_level(tokens, n)
                levels.append({
                    "level":       n,
                    "label":       f"{n}-gram",
                    "probability": round(level_probs[n] * 100, 1),
                    "count":       sum(cands.values()) if cands else 0,
                })

        # ── Stage-1 × Stage-2 combined: the true next-word selection dist. ────
        selection = self._selection_distribution(tokens, temperature)

        return {
            "unigram":      unigram_probs,
            "bigram":       bigram_probs,
            "trigram":      trigram_probs,
            "current_word": display_last,
            "context":      ctx_display,
            "levels":       levels,
            "selection":    selection,
            **higher,
        }
