"""
Train a SentencePiece BPE tokenizer on story.txt.

Pure BPE mode (no TR token placeholders):
  1. Stream story.txt, strip Wikipedia @-@ / @.@ / @,@ artefacts, lowercase.
  2. Write to a temp file (one line per original line).
  3. Train SentencePiece BPE with vocab_size=16000.
     Special ids: pad=0, unk=1, bos=2, eos=3.

Outputs:
  tokenizer/spm.model
  tokenizer/spm.vocab

Usage:
  python tokenizer/build_tokenizer.py
  python tokenizer/build_tokenizer.py --vocab_size 16000 --input story.txt
"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_utils import clean_wikipedia_artifacts

DEFAULT_INPUT = "story.txt"
DEFAULT_MODEL_PREFIX = "tokenizer/spm"
DEFAULT_VOCAB_SIZE = 16000
CHUNK_BYTES = 4 * 1024 * 1024  # 4 MB


def stream_preprocess(input_path: str, output_path: str) -> int:
    """Stream story.txt → cleaned text, preserving newlines for SPM's line-per-sentence input."""
    total_chars = 0
    with open(input_path, "r", encoding="utf-8", errors="replace") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        leftover = ""
        while True:
            chunk = fin.read(CHUNK_BYTES)
            if not chunk:
                break
            text = leftover + chunk
            cut = text.rfind("\n")
            if cut == -1:
                leftover = text
                continue
            head, leftover = text[:cut + 1], text[cut + 1:]
            cleaned = clean_wikipedia_artifacts(head).lower()
            fout.write(cleaned)
            total_chars += len(cleaned)
        if leftover:
            cleaned = clean_wikipedia_artifacts(leftover).lower()
            fout.write(cleaned)
            total_chars += len(cleaned)
    return total_chars


def train(input_path: str, model_prefix: str, vocab_size: int):
    import sentencepiece as spm

    os.makedirs(os.path.dirname(model_prefix) or ".", exist_ok=True)

    print(f"[1/3] Preprocessing: {input_path}")
    with tempfile.NamedTemporaryFile(
        prefix="spm_input_", suffix=".txt", delete=False, mode="w", encoding="utf-8"
    ) as tmp:
        tmp_path = tmp.name
    try:
        n_chars = stream_preprocess(input_path, tmp_path)
        print(f"      Chars written: {n_chars:,}")

        print(f"[2/3] Training SentencePiece BPE (vocab_size={vocab_size})")
        spm.SentencePieceTrainer.train(
            input=tmp_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            input_sentence_size=2_000_000,
            shuffle_input_sentence=True,
            num_threads=os.cpu_count() or 4,
            train_extremely_large_corpus=True,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    print(f"[3/3] Loading and verifying: {model_prefix}.model")
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    print(f"  OK — vocab size: {sp.get_piece_size()}")
    print(f"Model files: {model_prefix}.model, {model_prefix}.vocab")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--model_prefix", default=DEFAULT_MODEL_PREFIX)
    parser.add_argument("--vocab_size", type=int, default=DEFAULT_VOCAB_SIZE)
    args = parser.parse_args()
    train(args.input, args.model_prefix, args.vocab_size)


if __name__ == "__main__":
    main()
