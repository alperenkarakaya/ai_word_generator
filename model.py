"""
N-gram Language Model with Stupid Backoff (orders 1..max_n).
Primary text generation engine for the AI Text Generator.

Stupid Backoff: at generation time, try the highest-order n-gram that has
a matching context; fall back one order at a time until unigram.
"""
import os
import pickle
import random
import re
from collections import Counter, defaultdict

from text_utils import (
    full_clean,
    is_punctuation_token,
    is_sentence_ending,
    replace_punctuation_with_tokens,
    restore_punctuation_from_tokens,
)


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

    def _predict_next_with_backoff(self, context: list):
        """
        Stupid Backoff (stochastic): try max_n-gram first, fall back to (n-1)-gram,
        down to bigram, then unigram.  Uses weighted random sampling for variety.
        Used by generate/sentence-complete flows.
        """
        for n in range(self.max_n, 3, -1):
            if n not in self.ngram_counts:
                continue
            if len(context) < n - 1:
                continue
            key   = tuple(context[-(n - 1):])
            cands = self.ngram_counts[n].get(key)
            if cands:
                return random.choices(list(cands), weights=list(cands.values()), k=1)[0]

        if len(context) >= 2:
            key   = (context[-2], context[-1])
            cands = self.trigram_counts.get(key)
            if cands:
                return random.choices(list(cands), weights=list(cands.values()), k=1)[0]

        if context:
            cands = self.bigram_counts.get(context[-1])
            if cands:
                return random.choices(list(cands), weights=list(cands.values()), k=1)[0]

        if self.unigram_counts:
            top = self.unigram_counts.most_common(200)
            words, weights = zip(*top)
            return random.choices(words, weights=weights, k=1)[0]

        return None

    def _predict_most_likely_next(self, context: list):
        """
        Stupid Backoff (deterministic): same order of fallback but always picks
        the single most-frequent candidate.  Used for ghost text so the suggestion
        is stable while the user types.
        """
        for n in range(self.max_n, 3, -1):
            if n not in self.ngram_counts:
                continue
            if len(context) < n - 1:
                continue
            key   = tuple(context[-(n - 1):])
            cands = self.ngram_counts[n].get(key)
            if cands:
                return max(cands, key=cands.get)

        if len(context) >= 2:
            key   = (context[-2], context[-1])
            cands = self.trigram_counts.get(key)
            if cands:
                return max(cands, key=cands.get)

        if context:
            cands = self.bigram_counts.get(context[-1])
            if cands:
                return max(cands, key=cands.get)

        if self.unigram_counts:
            return self.unigram_counts.most_common(1)[0][0]

        return None

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

    def predict_until_sentence_end(self, text, max_words=50):
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

        for _ in range(max_words):
            nw = self._predict_next_with_backoff(context)
            if nw is None:
                break
            generated.append(nw)
            context.append(nw)
            if is_sentence_ending(nw):
                break

        result = " ".join(generated)
        if need_leading_space and result:
            result = " " + result
        return restore_punctuation_from_tokens(result)

    def predict_paragraph(self, text, max_sentences=5, max_words_per_sentence=50):
        """
        Chain predict_until_sentence_end to produce a multi-sentence paragraph.
        Each call uses the full accumulated text as context so later sentences
        are conditioned on earlier ones.
        """
        accumulated = text
        completions = []
        for _ in range(max_sentences):
            comp = self.predict_until_sentence_end(accumulated, max_words=max_words_per_sentence)
            if not comp.strip():
                break
            completions.append(comp)
            accumulated = accumulated + comp
        return "".join(completions)

    # ── ghost text (inline word suggestion) ───────────────────────────────────

    def get_ghost_text(self, text: str, max_ghost_words: int = 7) -> str:
        """
        Returns the full ghost-layer string showing up to max_ghost_words words ahead.

        Algorithm
        ---------
        Step 1 — partial-word resolution:
            If the cursor is inside an incomplete word (e.g. "ca"), find the
            most-frequent word that starts with that prefix ("cat") and make it
            the first ghost token.  The running context is updated so that step 2
            predicts *after* "cat", not after the raw prefix "ca".

        Step 2 — deterministic chain generation:
            Call _predict_most_likely_next(context) up to max_ghost_words times.
            Each new token is appended to `context` so word n+1 is conditioned on
            words 1..n (not just the original user text).  Stop early when a
            sentence-ending token (TR001 / TR003 / TR004) is produced — the
            period is *included* in the ghost so the user sees where the sentence ends.

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

        # ── Step 2: generate up to max_ghost_words tokens (deterministic) ────
        remaining = max_ghost_words - len(ghost_tokens)
        for _ in range(remaining):
            tok = self._predict_most_likely_next(context)
            if tok is None:
                break
            ghost_tokens.append(tok)
            context.append(tok)
            if is_sentence_ending(tok):
                break               # include the period, then stop

        if not ghost_tokens:
            return ""

        # ── Step 3: build ghost_full string ──────────────────────────────────
        display = self._tokens_to_text(ghost_tokens)

        if trailing_space:
            # "the cat |…"  → "the cat sat on the mat."
            return text + display
        elif last in self.word_counts:
            # "the cat|"    → "the cat sat on the mat."
            return text + " " + display
        else:
            # "the ca|"     → "the cat sat on the mat."
            last_space = text.rfind(" ")
            prefix = text[: last_space + 1] if last_space >= 0 else ""
            return prefix + display

    # ── probability panel (UI) ─────────────────────────────────────────────────

    def get_probabilities(self, text, use_tokens=True):
        """Return unigram / bigram / trigram probability tables for the UI panels."""
        tokens = self._prepare_for_lookup(text) if text.strip() else []

        total = sum(self.unigram_counts.values()) or 1
        unigram_probs = [
            {
                "word": restore_punctuation_from_tokens(w) if is_punctuation_token(w) else w,
                "probability": round(c / total * 100, 1),
            }
            for w, c in self.unigram_counts.most_common(10)
        ]

        if not tokens:
            return {
                "unigram": unigram_probs, "bigram": [], "trigram": [],
                "current_word": "", "context": "",
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
                    })
            w1 = restore_punctuation_from_tokens(tokens[-2]) if is_punctuation_token(tokens[-2]) else tokens[-2]
            w2 = restore_punctuation_from_tokens(tokens[-1]) if is_punctuation_token(tokens[-1]) else tokens[-1]
            ctx_display = f"{w1} {w2}"

        display_last = restore_punctuation_from_tokens(last) if is_punctuation_token(last) else last

        return {
            "unigram":      unigram_probs,
            "bigram":       bigram_probs,
            "trigram":      trigram_probs,
            "current_word": display_last,
            "context":      ctx_display,
        }
