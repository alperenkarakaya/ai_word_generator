"""
SentencePiece BPE tokenizer'ını eğitir.

Pipeline:
  1. story.txt'i akışla oku, Wikipedia @-@ artıklarını temizle.
  2. Noktalamaları TR001..TR017 tokenlerine çevir.
  3. Geçici düz-metin dosyasını SentencePiece girişi olarak yaz.
  4. SentencePiece BPE eğit (vocab_size=16000, TR* tokenlar user_defined_symbols).
  5. Tüm TR* tokenlerin tek id'ye düştüğünü doğrula.

Çıktılar:
  tokenizer/spm.model
  tokenizer/spm.vocab

Kullanım:
  python tokenizer/build_tokenizer.py
  python tokenizer/build_tokenizer.py --vocab_size 16000 --input story.txt
"""
import argparse
import os
import sys
import tempfile

# Bu modül komut satırından çağrıldığında repo kökünü import yoluna ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_utils import (
    ALL_TR_TOKENS,
    clean_wikipedia_artifacts,
    replace_punctuation_with_tokens,
)

DEFAULT_INPUT = "story.txt"
DEFAULT_MODEL_PREFIX = "tokenizer/spm"
DEFAULT_VOCAB_SIZE = 16000
CHUNK_BYTES = 4 * 1024 * 1024  # 4 MB


def stream_preprocess(input_path: str, output_path: str) -> int:
    """story.txt'i akışla okuyup ön-işlenmiş halini output_path'e yaz."""
    total_chars = 0
    with open(input_path, "r", encoding="utf-8", errors="replace") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        leftover = ""
        while True:
            chunk = fin.read(CHUNK_BYTES)
            if not chunk:
                break
            text = leftover + chunk
            # Satır sınırında kes ki TR017 (satır sonu) doğru yerleşsin.
            cut = text.rfind("\n")
            if cut == -1:
                leftover = text
                continue
            head, leftover = text[:cut + 1], text[cut + 1:]
            cleaned = clean_wikipedia_artifacts(head)
            tokenized = replace_punctuation_with_tokens(cleaned)
            # TR017 replaced \n, so put real newlines back so SPM reads line-by-line
            tokenized = tokenized.replace('TR017 ', 'TR017\n')
            fout.write(tokenized)
            total_chars += len(tokenized)
        if leftover:
            cleaned = clean_wikipedia_artifacts(leftover)
            tokenized = replace_punctuation_with_tokens(cleaned)
            tokenized = tokenized.replace('TR017 ', 'TR017\n')
            fout.write(tokenized)
            total_chars += len(tokenized)
    return total_chars


def train(input_path: str, model_prefix: str, vocab_size: int):
    import sentencepiece as spm

    os.makedirs(os.path.dirname(model_prefix) or ".", exist_ok=True)

    print(f"[1/4] Ön-işleme: {input_path}")
    with tempfile.NamedTemporaryFile(
        prefix="spm_input_", suffix=".txt", delete=False, mode="w", encoding="utf-8"
    ) as tmp:
        tmp_path = tmp.name
    try:
        n_chars = stream_preprocess(input_path, tmp_path)
        print(f"      Yazılan karakter: {n_chars:,}")

        print(f"[2/4] SentencePiece BPE eğitimi (vocab_size={vocab_size})")
        spm.SentencePieceTrainer.train(
            input=tmp_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            user_defined_symbols=ALL_TR_TOKENS,
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

    print(f"[3/4] Doğrulama: {model_prefix}.model")
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")

    print(f"[4/4] TR tokenlerin tek id olduğunu doğrula")
    bad = []
    for tok in ALL_TR_TOKENS:
        ids = sp.encode(tok, out_type=int)
        if len(ids) != 1:
            bad.append((tok, ids))
    if bad:
        for tok, ids in bad:
            print(f"  HATA: {tok} -> {ids}")
        raise SystemExit("TR tokenleri tek id'ye düşmedi. user_defined_symbols ayarını kontrol edin.")
    print(f"  OK — {len(ALL_TR_TOKENS)} TR tokenin tamamı tek id.")
    print(f"\nVocab boyutu: {sp.get_piece_size()}")
    print(f"Model dosyaları: {model_prefix}.model, {model_prefix}.vocab")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Ham metin dosyası (story.txt)")
    parser.add_argument("--model_prefix", default=DEFAULT_MODEL_PREFIX, help="Çıktı dosya öneki")
    parser.add_argument("--vocab_size", type=int, default=DEFAULT_VOCAB_SIZE)
    args = parser.parse_args()
    train(args.input, args.model_prefix, args.vocab_size)


if __name__ == "__main__":
    main()
