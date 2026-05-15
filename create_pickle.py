"""
Build the N-gram model pickle from story.txt.
Run this once (or whenever you want to rebuild):  python create_pickle.py
"""
import pickle
import re
import os
import text_utils
from model import NgramPredictor

# ── settings ──────────────────────────────────────────────────────────────────
MAX_N      = 7           # highest n-gram order (Stupid Backoff goes from 7 down to 1)
READ_LIMIT = 10_000_000  # characters to read (~1.5M words); 0 = full file

INPUT_FILE   = "story.txt"
TEMP_FILE    = "story_tokenized_temp.txt"
OUTPUT_FILE  = "saved_model.pkl"


def preprocess_and_tokenize(text: str) -> str:
    """
    Clean Wikipedia artefacts, lowercase, and replace punctuation with TR tokens.
    Must match the transformation applied in _prepare_for_lookup at inference time.
    """
    # Wikipedia-specific patterns
    text = re.sub(r"@\s*-\s*@", "-", text)
    text = re.sub(r"@\s*\.\s*@", ".", text)
    text = re.sub(r"@\s*,\s*@", ",", text)
    text = text.replace("@", "")

    # Lowercase so model tokens match the lowercased lookup at inference time
    text = text.lower()

    # Replace punctuation with TR tokens
    mapping     = text_utils.PUNCTUATION_TOKENS
    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
    pattern     = "|".join(map(re.escape, sorted_keys))

    def replace_func(m):
        return f" {mapping[m.group(0)]} "

    text = re.sub(pattern, replace_func, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── main ──────────────────────────────────────────────────────────────────────

print(f"1. Reading '{INPUT_FILE}'...")
try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw = f.read(READ_LIMIT) if READ_LIMIT > 0 else f.read()

    chars_read = len(raw)
    print(f"   Read {chars_read:,} characters.")

    print("2. Tokenising (TR tokens)...")
    tokenized = preprocess_and_tokenize(raw)

    print(f"3. Writing temp file '{TEMP_FILE}'...")
    with open(TEMP_FILE, "w", encoding="utf-8") as f:
        f.write(tokenized)

    print(f"4. Training {MAX_N}-gram model (Stupid Backoff)...")
    model = NgramPredictor(max_n=MAX_N)
    model.train_from_file(TEMP_FILE)

    print(f"5. Saving model to '{OUTPUT_FILE}'...")
    ngram_counts_serialisable = {
        n: {k: dict(v) for k, v in table.items()}
        for n, table in model.ngram_counts.items()
    }
    model_data = {
        "word_counts":    model.word_counts,
        "unigram_counts": model.unigram_counts,
        "bigram_counts":  dict(model.bigram_counts),
        "trigram_counts": dict(model.trigram_counts),
        "ngram_counts":   ngram_counts_serialisable,
        "max_n":          model.max_n,
        "total_words":    model.total_words,
    }
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(model_data, f)

    print()
    print("=" * 50)
    print("  Model saved successfully!")
    print(f"  max_n = {MAX_N}  (Stupid Backoff 7-gram -> 1-gram)")
    print(f"  Vocab : {len(model.word_counts):,} unique tokens")
    print(f"  Run 'python app.py' to start the server.")
    print("=" * 50)

except FileNotFoundError:
    print(f"ERROR: '{INPUT_FILE}' not found.")
except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()
