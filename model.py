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
        Stupid Backoff: try max_n-gram first, fall back to (n-1)-gram,
        down to bigram, then unigram.
        Returns the sampled next token or None.
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
