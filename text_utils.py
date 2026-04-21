import os
import re
import unicodedata

# Noktalama token mapping
PUNCTUATION_TOKENS = {
    '.': 'TR001', ',': 'TR002', '!': 'TR003', '?': 'TR004',
    ';': 'TR005', ':': 'TR006', '-': 'TR007', '(': 'TR010',
    ')': 'TR011', '"': 'TR012', "'": 'TR013', '...': 'TR016',
    '\n': 'TR017'
}
TOKEN_TO_PUNCTUATION = {v: k for k, v in PUNCTUATION_TOKENS.items()}

# Tüm TR* tokenleri (SentencePiece user_defined_symbols için)
ALL_TR_TOKENS = sorted(PUNCTUATION_TOKENS.values())

SENTENCE_END_TOKENS = ('TR001', 'TR003', 'TR004')
PARAGRAPH_BREAK_TOKENS = ('TR017',)

def replace_punctuation_with_tokens(text: str) -> str:
    text = text.replace('...', 'TR016 ')
    for punct, token in sorted(PUNCTUATION_TOKENS.items(), key=lambda x: -len(x[0])):
        if punct == '...': continue
        text = text.replace(punct, f'{token} ')
    return text

def restore_punctuation_from_tokens(text: str) -> str:
    for token, punct in TOKEN_TO_PUNCTUATION.items():
        text = text.replace(f' {token} ', punct + ' ')
        text = text.replace(f'{token} ', punct + ' ')
        if text.endswith(token):
            text = text[:-len(token)] + punct
    return text

def full_clean(text: str, lowercase: bool = True, use_tokens: bool = True) -> str:
    # Evrensel normalizasyon (Türkçe/İngilizce fark etmez)
    text = unicodedata.normalize('NFC', text)
    if lowercase: text = text.lower()
    if use_tokens: text = replace_punctuation_with_tokens(text)
    # Gereksiz boşlukları temizle
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_punctuation_token(token: str) -> bool:
    return token.startswith('TR') and len(token) == 5 and token[2:].isdigit()

def is_sentence_ending(token: str) -> bool:
    return token in SENTENCE_END_TOKENS

def is_paragraph_break(token: str) -> bool:
    return token in PARAGRAPH_BREAK_TOKENS


# --- Wikipedia ön-temizleme ---
# story.txt içindeki @-@, @.@, @,@ gibi WikiText kalıntılarını temizler.
_WIKI_PATTERNS = [
    (re.compile(r'@\s*-\s*@'), '-'),
    (re.compile(r'@\s*\.\s*@'), '.'),
    (re.compile(r'@\s*,\s*@'), ','),
]

def clean_wikipedia_artifacts(text: str) -> str:
    for pattern, repl in _WIKI_PATTERNS:
        text = pattern.sub(repl, text)
    return text.replace('@', '')


# --- SentencePiece sarmalayıcıları (Phase 1) ---
# build_tokenizer.py burada yazılan modeli üretir, encode_corpus.py ve
# transformer/sample.py burada açılan helper'ları çağırır.

_SP_PROCESSOR = None
_SP_MODEL_PATH = None


def load_tokenizer(model_path: str):
    """SentencePiece modelini yükle (idempotent)."""
    global _SP_PROCESSOR, _SP_MODEL_PATH
    if _SP_PROCESSOR is not None and _SP_MODEL_PATH == model_path:
        return _SP_PROCESSOR
    import sentencepiece as spm
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Tokenizer modeli bulunamadı: {model_path}")
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    _SP_PROCESSOR = sp
    _SP_MODEL_PATH = model_path
    return sp


def get_tokenizer():
    """Yüklenmiş SentencePiece nesnesini döndür (yoksa hata)."""
    if _SP_PROCESSOR is None:
        raise RuntimeError(
            "Tokenizer yüklenmemiş. Önce text_utils.load_tokenizer(path) çağırın."
        )
    return _SP_PROCESSOR


def encode(text: str, add_bos: bool = False, add_eos: bool = False):
    """Metni id listesine çevir. Önce TR tokenları yerleştirilir."""
    sp = get_tokenizer()
    cleaned = full_clean(text, lowercase=True, use_tokens=True)
    ids = sp.encode(cleaned, out_type=int)
    if add_bos and sp.bos_id() >= 0:
        ids = [sp.bos_id()] + ids
    if add_eos and sp.eos_id() >= 0:
        ids = ids + [sp.eos_id()]
    return ids


def decode(ids) -> str:
    """Id listesini metne çevir ve TR tokenlarını noktalamaya geri döndür."""
    sp = get_tokenizer()
    raw = sp.decode(list(ids))
    return restore_punctuation_from_tokens(raw)


def tr_token_ids(model_path: str | None = None):
    """SentencePiece içindeki her TR* tokenin id'sini döndür."""
    sp = load_tokenizer(model_path) if model_path else get_tokenizer()
    out = {}
    for tok in ALL_TR_TOKENS:
        ids = sp.encode(tok, out_type=int)
        # Tek id'ye düşmeli — değilse vocab build hatalı.
        if len(ids) != 1:
            raise ValueError(
                f"TR tokeni tek id'ye düşmedi: {tok} -> {ids}. "
                "build_tokenizer.py user_defined_symbols ayarını kontrol edin."
            )
        out[tok] = ids[0]
    return out