# Project Init File

Claude reads this file at the start of every session. It holds project-specific
standing instructions, architecture notes, and a running log of major changes.
Update it when the project's shape or conventions shift.

---

## Standing instructions for Claude

- **Auto-push after major changes.** When a major change is made to the project
  or when we're starting a new phase / trying out a new component, push to
  GitHub without asking first. Examples that qualify:
  - Rewriting a core pipeline stage (tokenizer, encoder, trainer, sampler)
  - Fixing a blocking bug that the user is about to test on Colab
  - Adding or removing a major feature
  Small edits (formatting, comments, single-line fixes) don't require push.
  Still NEVER push force, NEVER push to non-main branches without asking,
  NEVER push files that contain secrets.
- **Keep this file current.** Whenever the project direction changes, a major
  component is added/removed, or a new convention is adopted, append or update
  the relevant section below.

---

## What this project is

Graduation thesis ("üni bitirme") — a Turkish/English text generator trained
from scratch on an English Wikipedia dump. Phased roadmap:

1. **Words** — next-word prediction (legacy, done, N-gram + char-LSTM).
2. **Sentences** — decoder-only Transformer (← **current phase**).
3. **Paragraphs** — same model, multi-sentence generation.
4. **Content** — topic-conditioned generation.
5. **Optional Q&A** — instruction-style responses.

Flask UI (`app.py`) stays; it exposes both the legacy N-gram stats panels and
the new Transformer generation endpoints.

---

## Current architecture (Phase 2 — sentences)

**Pipeline (pure BPE, no TR token placeholders):**

```
story.txt (514 MB English Wiki)
    │
    ▼
tokenizer/build_tokenizer.py      # SentencePiece BPE, vocab=16000
    │
    ▼
tokenizer/spm.model, spm.vocab
    │
    ▼
tokenizer/encode_corpus.py        # corpus → data/train.bin, data/val.bin (uint16)
    │
    ▼
transformer/train.py              # decoder-only GPT, AdamW + cosine LR, AMP
    │
    ▼
checkpoints/transformer.pt        # best val-loss checkpoint
    │
    ▼
transformer/sample.py             # TransformerEngine — sentence/paragraph generation
    │
    ▼
app.py /predict_sentence, /predict_paragraph endpoints
```

**Key files:**
- `text_utils.py` — shared helpers (BPE load/encode/decode, Wikipedia clean).
  The bottom section of the file contains legacy TR-token compatibility shims
  used only by `model.py` (N-gram stats panels). New transformer code does NOT
  use these.
- `transformer/model.py` — GPT config + module.
- `transformer/dataset.py` — mmap'd BinDataset.
- `transformer/train.py` — training loop.
- `transformer/sample.py` — generation (KV-cache + sliding window + top-k/top-p).
- `tokenizer/build_tokenizer.py` — trains SPM BPE on raw cleaned text.
- `tokenizer/encode_corpus.py` — corpus → uint16 bin files.
- `notebooks/colab_train.ipynb` — end-to-end Colab recipe.
- `model.py` + `create_pickle.py` — legacy N-gram (will be removed after
  sentence phase is stable).

**Training environment:** Colab / Kaggle GPU (T4 baseline, A100 if available).
Checkpoints and tokenizer are mirrored to `/content/drive/MyDrive/uni_bitirme/`
so Colab session expiry doesn't lose work.

---

## Changelog (major changes only)

Append new entries at the top. Format: `YYYY-MM-DD — short description`.

- **2026-04-22** — Switched to pure BPE. Removed TR001..TR017 punctuation
  placeholder system from the training / inference pipeline. SentencePiece
  now handles punctuation natively; sentence-end detection happens on the
  decoded string (checks for `.`, `!`, `?`). Legacy TR helpers kept in
  `text_utils.py` only for `model.py`'s N-gram stats panels.
- **2026-04-22** — Fixed SPM trainer crash: preprocessing was collapsing all
  newlines into `TR017 ` so SPM saw a single giant line and filtered it as
  empty. (Later superseded by the pure-BPE switch above.)
- **Earlier** — Replaced N-gram/char-LSTM core with decoder-only Transformer
  skeleton (commit 46cbefe, "Phase 1: Replace N-gram/char-LSTM with
  decoder-only Transformer").

---

## Notes for future Claude

- The user is a CS senior; explain deeply when proposing alternatives rather
  than just executing (this is also in auto-memory).
- Flask UI is load-bearing for the thesis demo — don't break `app.py` without
  flagging it.
- `story.txt` is 514 MB and gitignored. It lives on the user's machine and
  on Drive under `uni_bitirme/`.
- `checkpoints/*.pt`, `tokenizer/spm.*`, `data/*.bin` are all gitignored.
  If you reorganize outputs, update `.gitignore`.
