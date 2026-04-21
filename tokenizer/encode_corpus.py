"""
story.txt'i tokenize edip uint16 .bin dosyalarına yazar.

Çıktılar (varsayılan):
  data/train.bin  — vocab id'lerinin uint16 ardışık dizisi (95% korpus)
  data/val.bin    — son %5 doğrulama parçası

Bu format nanoGPT-tarzı bir mmap-dataset için yeterlidir; her batch
rastgele bir başlangıç indexi seçip block_size+1 token okur.

Kullanım:
  python tokenizer/encode_corpus.py
  python tokenizer/encode_corpus.py --input story.txt --val_fraction 0.05
"""
import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_utils import (
    clean_wikipedia_artifacts,
    load_tokenizer,
    replace_punctuation_with_tokens,
)

DEFAULT_INPUT = "story.txt"
DEFAULT_TOKENIZER = "tokenizer/spm.model"
DEFAULT_OUT_DIR = "data"
CHUNK_BYTES = 4 * 1024 * 1024  # 4 MB — bellek dostu okuma


def iter_encoded_chunks(input_path: str, sp):
    """story.txt'i akışla okuyup id listeleri üret."""
    with open(input_path, "r", encoding="utf-8", errors="replace") as fin:
        leftover = ""
        with tqdm(unit="MB", desc="encode") as pbar:
            while True:
                chunk = fin.read(CHUNK_BYTES)
                if not chunk:
                    break
                pbar.update(len(chunk) // (1024 * 1024))
                text = leftover + chunk
                cut = text.rfind("\n")
                if cut == -1:
                    leftover = text
                    continue
                head, leftover = text[:cut + 1], text[cut + 1:]
                cleaned = clean_wikipedia_artifacts(head).lower()
                tokenized = replace_punctuation_with_tokens(cleaned)
                ids = sp.encode(tokenized, out_type=int)
                yield np.asarray(ids, dtype=np.uint16)
            if leftover:
                cleaned = clean_wikipedia_artifacts(leftover).lower()
                tokenized = replace_punctuation_with_tokens(cleaned)
                ids = sp.encode(tokenized, out_type=int)
                yield np.asarray(ids, dtype=np.uint16)


def encode(input_path: str, tokenizer_path: str, out_dir: str, val_fraction: float):
    sp = load_tokenizer(tokenizer_path)
    if sp.get_piece_size() > 65535:
        raise ValueError(
            f"vocab_size={sp.get_piece_size()} uint16 limitini aşıyor. "
            "encode_corpus.py'yi uint32'ye geçirin veya vocab'ı küçültün."
        )

    os.makedirs(out_dir, exist_ok=True)
    tmp_path = os.path.join(out_dir, "all.bin.tmp")

    total_tokens = 0
    with open(tmp_path, "wb") as fout:
        for arr in iter_encoded_chunks(input_path, sp):
            fout.write(arr.tobytes())
            total_tokens += arr.size
    print(f"Toplam token: {total_tokens:,}")

    # train/val ayrımı: dosyanın sonundan val_fraction kadarını val'e koy.
    val_tokens = int(total_tokens * val_fraction)
    train_tokens = total_tokens - val_tokens
    print(f"Train: {train_tokens:,}  Val: {val_tokens:,}")

    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")

    with open(tmp_path, "rb") as fin:
        with open(train_path, "wb") as fout:
            remaining = train_tokens * 2  # uint16 = 2 bayt
            while remaining > 0:
                buf = fin.read(min(8 * 1024 * 1024, remaining))
                if not buf:
                    break
                fout.write(buf)
                remaining -= len(buf)
        with open(val_path, "wb") as fout:
            while True:
                buf = fin.read(8 * 1024 * 1024)
                if not buf:
                    break
                fout.write(buf)

    os.unlink(tmp_path)
    print(f"Yazıldı: {train_path}, {val_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    args = parser.parse_args()
    encode(args.input, args.tokenizer, args.out_dir, args.val_fraction)


if __name__ == "__main__":
    main()
