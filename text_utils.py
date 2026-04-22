"""
Text utility helpers for the Transformer pipeline.

Pure BPE mode: we no longer replace punctuation with TR001..TR017 placeholders.
SentencePiece handles punctuation naturally — '.', ',', '!', '?' etc. each become
their own BPE piece. Sentence-end detection happens on the DECODED string.
"""
import os
import re
import unicodedata


SENTENCE_END_CHARS = (".", "!", "?")
PARAGRAPH_BREAK_CHAR = "\n"


def full_clean(text: str, lowercase: bool = True) -> str:
    """NFC normalize, optionally lowercase, collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    if lowercase:
        text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- Wikipedia pre-cleaning -------------------------------------------------
# story.txt contains WikiText artefacts like @-@, @.@, @,@. Strip them.
_WIKI_PATTERNS = [
    (re.compile(r"@\s*-\s*@"), "-"),
    (re.compile(r"@\s*\.\s*@"), "."),
    (re.compile(r"@\s*,\s*@"), ","),
]


def clean_wikipedia_artifacts(text: str) -> str:
    for pattern, repl in _WIKI_PATTERNS:
        text = pattern.sub(repl, text)
    return text.replace("@", "")


# --- SentencePiece helpers --------------------------------------------------
_SP_PROCESSOR = None
_SP_MODEL_PATH = None


def load_tokenizer(model_path: str):
    """Load a SentencePiece model (idempotent)."""
    global _SP_PROCESSOR, _SP_MODEL_PATH
    if _SP_PROCESSOR is not None and _SP_MODEL_PATH == model_path:
        return _SP_PROCESSOR
    import sentencepiece as spm
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Tokenizer not found: {model_path}")
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    _SP_PROCESSOR = sp
    _SP_MODEL_PATH = model_path
    return sp


def get_tokenizer():
    if _SP_PROCESSOR is None:
        raise RuntimeError("Tokenizer not loaded. Call text_utils.load_tokenizer(path) first.")
    return _SP_PROCESSOR


def encode(text: str, add_bos: bool = False, add_eos: bool = False):
    """Text → id list. Applies clean_wikipedia_artifacts + lowercase."""
    sp = get_tokenizer()
    cleaned = full_clean(clean_wikipedia_artifacts(text), lowercase=True)
    ids = sp.encode(cleaned, out_type=int)
    if add_bos and sp.bos_id() >= 0:
        ids = [sp.bos_id()] + ids
    if add_eos and sp.eos_id() >= 0:
        ids = ids + [sp.eos_id()]
    return ids


def decode(ids) -> str:
    sp = get_tokenizer()
    return sp.decode(list(ids))


def text_ends_sentence(text: str) -> bool:
    """True if the decoded text ends with '.', '!' or '?' (ignoring trailing whitespace)."""
    s = text.rstrip()
    return len(s) > 0 and s[-1] in SENTENCE_END_CHARS


# --- Legacy compatibility (only for the N-gram probability panels) ---------
# model.py / create_pickle.py still use the old TR token mapping for its stats
# view. Transformer training/inference no longer touches any of this.

PUNCTUATION_TOKENS = {
    ".": "TR001", ",": "TR002", "!": "TR003", "?": "TR004",
    ";": "TR005", ":": "TR006", "-": "TR007", "(": "TR010",
    ")": "TR011", '"': "TR012", "'": "TR013", "...": "TR016",
    "\n": "TR017",
}
TOKEN_TO_PUNCTUATION = {v: k for k, v in PUNCTUATION_TOKENS.items()}
ALL_TR_TOKENS = sorted(PUNCTUATION_TOKENS.values())
SENTENCE_END_TOKENS = ("TR001", "TR003", "TR004")
PARAGRAPH_BREAK_TOKENS = ("TR017",)


def replace_punctuation_with_tokens(text: str) -> str:
    text = text.replace("...", "TR016 ")
    for punct, token in sorted(PUNCTUATION_TOKENS.items(), key=lambda x: -len(x[0])):
        if punct == "...":
            continue
        text = text.replace(punct, f"{token} ")
    return text


def restore_punctuation_from_tokens(text: str) -> str:
    for token, punct in TOKEN_TO_PUNCTUATION.items():
        text = text.replace(f" {token} ", punct + " ")
        text = text.replace(f"{token} ", punct + " ")
        if text.endswith(token):
            text = text[:-len(token)] + punct
    return text


def is_punctuation_token(token: str) -> bool:
    return token.startswith("TR") and len(token) == 5 and token[2:].isdigit()


def is_sentence_ending(token: str) -> bool:
    return token in SENTENCE_END_TOKENS


def is_paragraph_break(token: str) -> bool:
    return token in PARAGRAPH_BREAK_TOKENS
