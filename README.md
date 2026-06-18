# AI Text Generator

Graduation thesis project — a web-based text generation system that demonstrates the
evolution from classical statistical language models (N-gram) to neural approaches
(Transformer), with live word-by-word autocomplete in the browser.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [N-gram Engine (Primary)](#n-gram-engine-primary)
   - [Stupid Backoff Algorithm](#stupid-backoff-algorithm)
   - [Ghost Text (Inline Autocomplete)](#ghost-text-inline-autocomplete)
   - [Sentence & Paragraph Generation](#sentence--paragraph-generation)
   - [Temperature Sampling](#temperature-sampling)
   - [Anti-Loop Mechanisms](#anti-loop-mechanisms)
   - [Punctuation Token System (TR Tokens)](#punctuation-token-system-tr-tokens)
4. [Transformer Engine (Alternative)](#transformer-engine-alternative)
   - [Model Architecture (GPT)](#model-architecture-gpt)
   - [KV-Cache & Sliding Window](#kv-cache--sliding-window)
   - [Sampling Parameters](#sampling-parameters)
   - [Sentence & Paragraph Stop Conditions](#sentence--paragraph-stop-conditions)
5. [SentencePiece BPE Tokenizer](#sentencepiece-bpe-tokenizer)
6. [Text Preprocessing Pipeline](#text-preprocessing-pipeline)
7. [Web UI](#web-ui)
   - [Editor & Keyboard Shortcuts](#editor--keyboard-shortcuts)
   - [Engine Selector & Temperature Slider](#engine-selector--temperature-slider)
   - [N-gram Probability Visualisation Panel](#n-gram-probability-visualisation-panel)
8. [Flask API Endpoints](#flask-api-endpoints)
9. [Project Structure](#project-structure)
10. [Setup & Installation](#setup--installation)
11. [Training Pipeline](#training-pipeline)
    - [N-gram Model Build](#n-gram-model-build)
    - [Tokenizer Training](#tokenizer-training)
    - [Corpus Encoding](#corpus-encoding)
    - [Transformer Training (Colab/Kaggle)](#transformer-training-colabkaggle)
12. [Configuration Reference](#configuration-reference)
13. [Developer](#developer)

---

## Overview

This project is a **Turkish Wikipedia** corpus-based text generation system built as a
Computer Science graduation thesis. It provides two generation engines:

| Engine | Type | Parameters | Requires GPU | Status |
|--------|------|------------|--------------|--------|
| **N-gram** | Statistical (7-gram Stupid Backoff) | ~60K vocabulary, 4-7 gram tables | No | Primary, always available |
| **Transformer** | Neural (decoder-only GPT) | ~30M trainable parameters | Yes (training only) | Alternative, trained on Colab |

Both engines are served through the same Flask web application with a shared editor UI.
The user can switch between engines in real-time via a dropdown selector.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Browser (index.html)                     │
│  ┌─────────────┐  ┌───────────────┐  ┌───────────────────┐  │
│  │  Text Editor │  │  Ghost Layer  │  │  Probability      │  │
│  │  (textarea)  │  │  (div overlay)│  │  Panel (7 cards)  │  │
│  └──────┬───────┘  └───────┬───────┘  └────────┬──────────┘  │
│         │ input event      │ /predict_next      │ /probs      │
└─────────┼──────────────────┼────────────────────┼────────────┘
          │                  │                    │
   ┌──────▼──────────────────▼────────────────────▼──────────┐
   │                     Flask API (app.py)                    │
   │  /predict_next  /predict_sentence  /predict_paragraph     │
   │  /probabilities /health  /                                │
   └──────┬───────────────────────────────┬──────────────────┘
          │                               │
   ┌──────▼──────────┐           ┌────────▼───────────────┐
   │  NgramPredictor  │           │  TransformerEngine     │
   │  (model.py)      │           │  (transformer/sample.py)│
   │                  │           │                         │
   │  saved_model.pkl │           │  checkpoints/           │
   │  (pickle)        │           │  transformer.pt         │
   └──────────────────┘           │  tokenizer/spm.model    │
                                  └─────────────────────────┘
```

---

## N-gram Engine (Primary)

**File:** `model.py` — class `NgramPredictor`

The N-gram engine is the core generation system. It builds frequency tables for word
sequences of length 1 through 7 from the training corpus, then uses Stupid Backoff at
generation time to pick the next word.

### Data Structures

| Attribute | Type | Description |
|-----------|------|-------------|
| `word_counts` | `Counter` | Total frequency of each token in the corpus |
| `unigram_counts` | `Counter` | Same as word_counts (1-gram frequencies) |
| `bigram_counts` | `defaultdict(Counter)` | `word → {next_word: count}` |
| `trigram_counts` | `defaultdict(Counter)` | `(w1, w2) → {w3: count}` |
| `ngram_counts` | `dict[int, defaultdict(Counter)]` | Orders 4-7: `{n: {(w1..wn-1): {wn: count}}}` |
| `total_words` | `int` | Total token count in training data |
| `max_n` | `int` | Maximum n-gram order (default: 7) |

### Hierarchical Sampling (Generation)

Sentence and paragraph generation no longer use deterministic backoff. Instead they use a
two-stage hierarchical sampler (`_sample_token`) built on a single tempered operator:

```
τ(x ; T)_i = x_i^(1/T) / Σ_j x_j^(1/T)      # T→0: argmax · T=1: proportional · T→∞: uniform

sample_token(context, T):
    # Stage 1 — sample which n-gram order to use
    for n in available orders:
        C_n      = total evidence at this context's n-gram entry
        score_n  = (n ** GAMMA) * (1 - exp(-C_n / KAPPA))   # favour high, well-supported orders
    n* ~ τ(score ; T)

    # Stage 2 — sample a word within the chosen order
    topK     = top-5 candidates of order n* by count
    score_i  = count_i / repetition_penalty(word_i)
    return  w* ~ τ(score ; T)
```

`GAMMA` (default 1.3) controls how strongly higher orders are preferred; `KAPPA` (default 3)
suppresses high-order contexts seen only once or twice (prevents verbatim memorised output).
Both stages share the temperature `T`: low `T` ≈ the old "always highest order + argmax"
behaviour, high `T` flattens both the order distribution and the word distribution. A
repetition penalty plus a bigram anti-loop guard keep generation coherent.

Ghost text uses a **separate, deterministic** path (it is a stable UI guidance feature):

| Method | Selection Strategy | Used By |
|--------|--------------------|---------|
| `_sample_token()` | Hierarchical level + top-K sampling (temperature-scaled) | Sentence/paragraph generation |
| `_predict_most_likely_next()` | Deterministic argmax (always picks most frequent) | Ghost text (all temperatures) |
| `_predict_ghost_token()` | Deterministic argmax with repetition blocking via `seen` set | Ghost text anti-loop chain |

### Ghost Text (Inline Autocomplete)

**Method:** `get_ghost_text(text, max_ghost_words=7)`

Ghost text is the gray inline suggestion that appears as the user types. It operates in
three steps:

**Step 1 — Partial word resolution:**
If the cursor is mid-word (e.g. `"ca"`), find the most frequent word starting with that
prefix (`"cat"`) and make it the first ghost token. The context is updated so Step 2
predicts after `"cat"`, not `"ca"`.

**Step 2 — Chain generation (up to 7 tokens), deterministic & temperature-free:**

Ghost is intentionally deterministic so the same editor state always yields the same
suggestion (stable, repeatable, demo-safe). It uses `_predict_most_likely_next()` with a
bigram-based anti-loop guard: before generating, all bigrams from the user's typed text are
recorded; if the predicted next token would form a bigram already seen, the model falls back
to `_predict_ghost_token()` which skips that token. This prevents ghost loops like
`"the first time in his career, the first time in his career..."`. Temperature affects only
sentence/paragraph **generation** (`_sample_token`), never the ghost suggestion.

**Step 3 — Assemble:**
Ghost tokens are converted to display text via `_tokens_to_text()`, which attaches
punctuation tokens (TR001 etc.) directly to the preceding word without a space. The full
ghost-layer string (user text + suggestion) is returned. The JS frontend slices at
`len(user_text)` to color the suggestion part gray.

**Return value examples:**

| User types | Ghost returns | Displayed suggestion |
|------------|---------------|---------------------|
| `"the cat "` | `"the cat sat on the mat."` | `sat on the mat.` (gray) |
| `"the cat"` | `"the cat sat on the mat."` | ` sat on the mat.` (gray) |
| `"the ca"` | `"the cat sat on the mat."` | `t sat on the mat.` (gray) |

### Sentence & Paragraph Generation

**`predict_until_sentence_end(text, max_words=50, temperature=1.0)`**

Generates tokens using Stupid Backoff until a sentence-ending punctuation token
(TR001=`.`, TR003=`!`, TR004=`?`) is produced or `max_words` is reached.

- If the cursor is mid-word, it first completes the partial word (same prefix matching
  as ghost text).
- If the cursor is at the end of a complete word, the result starts with a space so
  words don't run together.
- Returns a string suffix to append to the editor text.

**`predict_paragraph(text, max_sentences=5, max_words_per_sentence=50, temperature=1.0)`**

Chains `predict_until_sentence_end` up to `max_sentences` times. Each call uses the full
accumulated text as context, so later sentences are conditioned on earlier ones.

### Temperature Sampling

**Method:** `_sample_with_temperature(candidates, temperature)`

Temperature controls output diversity by re-weighting the probability distribution:

```python
weights = [count ** (1.0 / temperature) for count in candidates.values()]
next_word = random.choices(words, weights=weights, k=1)[0]
```

| Temperature | Formula Effect | Behavioral Effect |
|-------------|----------------|-------------------|
| **0.5** | `count^2.0` — squares the counts | Very focused, top word dominates |
| **1.0** | `count^1.0` — unchanged | Default behavior (original frequencies) |
| **1.5** | `count^0.67` — compresses differences | More uniform, more variety |
| **2.0** | `count^0.5` — square root of counts | Significantly flattened |
| **2.5** | `count^0.4` — even more compressed | Near-uniform, maximum variety |

**Temperature range in UI:** 0.5 to 2.5 (step 0.1)

**Minimum candidates rule:** When temperature > 1.0, the backoff requires **at least 2
candidates** (`min_cands=2`) before committing to a match. A single-candidate entry at
7-gram level is just memorized text, and temperature can't add variety to a set of one.
Falling back forces the model to reach a level (usually trigram/bigram) where multiple
next words are genuinely observed.

### Anti-Loop Mechanisms

The N-gram model has three anti-loop safeguards:

1. **Bigram seen-set (ghost text, T=1.0):** Before generating ghost tokens, all bigrams
   from the user's existing text are recorded. If the next predicted token would recreate
   an already-seen bigram, the model switches to `_predict_ghost_token()` which skips
   that token and tries the next-best candidate.

2. **`_predict_ghost_token()` soft block:** This method filters out tokens in the `seen`
   set from the candidate pool. If filtering leaves an empty pool, it falls back to the
   unfiltered candidates (punctuation tokens are never added to `seen`, so they can
   legitimately recur).

3. **min_cands ≥ 2 (generation, T>1.0):** Forces fallback past memorized single-candidate
   entries to levels with genuine choice.

### Punctuation Token System (TR Tokens)

The N-gram engine replaces punctuation with placeholder tokens during training and
inference. This ensures punctuation is treated as separate tokens:

| Punctuation | Token | Description |
|-------------|-------|-------------|
| `.` | `TR001` | Period (sentence end) |
| `,` | `TR002` | Comma |
| `!` | `TR003` | Exclamation (sentence end) |
| `?` | `TR004` | Question mark (sentence end) |
| `;` | `TR005` | Semicolon |
| `:` | `TR006` | Colon |
| `-` | `TR007` | Hyphen |
| `(` | `TR010` | Open parenthesis |
| `)` | `TR011` | Close parenthesis |
| `"` | `TR012` | Double quote |
| `'` | `TR013` | Single quote / apostrophe |
| `...` | `TR016` | Ellipsis |
| `\n` | `TR017` | Newline (paragraph break) |

**Sentence-ending tokens:** TR001, TR003, TR004
**Paragraph-break tokens:** TR017

**`_tokens_to_text(tokens)`** converts model tokens back to display text by attaching
punctuation tokens directly to the preceding word:
- `["cat", "sat", "TR001"]` → `"cat sat."`
- `["TR011", "TR002", "known", "as"]` → `"), known as"`

---

## Transformer Engine (Alternative)

**Files:** `transformer/model.py`, `transformer/sample.py`, `transformer/train.py`, `transformer/dataset.py`

### Model Architecture (GPT)

A decoder-only GPT (Generative Pre-trained Transformer) with the following architecture:

```
Input Token IDs
       │
       ▼
┌──────────────────┐
│  Token Embedding  │  (vocab_size × d_model)
│  + Positional     │  (block_size × d_model, learnable)
│    Embedding      │
└────────┬─────────┘
         │ + Dropout(0.1)
         ▼
┌──────────────────────────────────────────┐
│  Block × 6  (Pre-Norm architecture)      │
│                                          │
│  ┌─────────────────────────────────────┐ │
│  │ LayerNorm                           │ │
│  │ Causal Multi-Head Self-Attention    │ │
│  │   (6 heads, head_dim=64)           │ │
│  │ + Residual connection              │ │
│  ├─────────────────────────────────────┤ │
│  │ LayerNorm                           │ │
│  │ FFN: Linear(384→1536) → GELU       │ │
│  │      → Linear(1536→384) → Dropout  │ │
│  │ + Residual connection              │ │
│  └─────────────────────────────────────┘ │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│  Final LayerNorm  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Output Head      │  (d_model → vocab_size)
│  (Weight Tying:   │   shares weights with Token Embedding)
│   head.weight =   │
│   tok_emb.weight) │
└──────────────────┘
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `vocab_size` | 16,000 | BPE vocabulary (SentencePiece) |
| `block_size` | 256 | Maximum context window (tokens) |
| `n_layer` | 6 | Number of transformer blocks |
| `n_head` | 6 | Number of attention heads |
| `d_model` | 384 | Model dimension / embedding size |
| `d_ff` | 1,536 | Feed-forward inner dimension (4× d_model) |
| `head_dim` | 64 | Per-head dimension (d_model / n_head) |
| `dropout` | 0.1 | Dropout rate (attention + FFN + embedding) |
| `pad_id` | 0 | Padding token ID (ignored in loss) |
| **Total parameters** | **~30M** | Trainable parameters |

### Key Implementation Details

- **Pre-Norm:** LayerNorm is applied **before** attention and FFN (not after), following
  GPT-2 and later conventions. This improves training stability.
- **Weight Tying:** The output projection head shares weights with the token embedding
  layer (`head.weight = tok_emb.weight`). This reduces parameters and improves
  generalization.
- **Scaled Dot-Product Attention:** Uses PyTorch 2.x's built-in
  `F.scaled_dot_product_attention` with causal masking. Automatically uses FlashAttention
  when available on GPU.
- **Weight Initialization:** All linear layers use `N(0, 0.02)`, biases are zeroed,
  LayerNorm weights are set to 1.0.

### KV-Cache & Sliding Window

**File:** `transformer/sample.py` — class `TransformerEngine`

Generation uses a KV-cache to avoid recomputing attention for previously generated tokens:

1. **First pass:** The full prompt is processed, producing KV-cache entries for each layer.
2. **Subsequent steps:** Only the newly generated token is fed through the model. The
   KV-cache from previous steps is concatenated with the new K/V.
3. **Sliding window fallback:** When `n_cached >= block_size` (256 tokens), the cache is
   discarded and the last `block_size` tokens are re-processed from scratch. This
   prevents position embedding overflow.

### Sampling Parameters

**Dataclass:** `SamplingConfig`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_new_tokens` | 80 | Maximum tokens to generate |
| `temperature` | 0.85 | Softmax temperature (lower = more focused) |
| `top_k` | 40 | Keep only top-k logits before sampling |
| `top_p` | 0.92 | Nucleus sampling — keep smallest set of tokens with cumulative probability ≥ top_p |
| `repetition_penalty` | 1.15 | Penalize previously generated tokens (>1.0 = penalize) |

**Sampling pipeline** (applied in order):

1. **Repetition penalty:** For each unique token already in the history, if the logit
   is positive, divide by `repetition_penalty`; if negative, multiply.
2. **Temperature scaling:** Divide all logits by temperature.
3. **Top-k filtering:** Zero out all logits below the k-th highest value.
4. **Top-p (nucleus) filtering:** Sort logits descending, compute cumulative softmax
   probabilities, zero out everything past the `top_p` threshold.
5. **Softmax + multinomial sampling.**

### Sentence & Paragraph Stop Conditions

Generation stop conditions are checked on the **decoded text** (not token IDs), since
SentencePiece BPE handles punctuation naturally:

| Method | Stop Condition |
|--------|----------------|
| `generate()` | Only `max_new_tokens` |
| `generate_until_sentence_end()` | Decoded text ends with `.`, `!`, or `?` |
| `generate_paragraph()` | `max_sentences` sentence-enders reached, OR newline (`\n`) in decoded text |

Paragraph generation uses slightly different defaults: `temperature=0.75`, `top_p=0.90`.

---

## SentencePiece BPE Tokenizer

**Files:** `tokenizer/build_tokenizer.py`, `tokenizer/encode_corpus.py`, `tokenizer/spm.model`

The Transformer engine uses a BPE (Byte Pair Encoding) tokenizer trained by SentencePiece
on the full Wikipedia corpus.

### Tokenizer Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | BPE (Byte Pair Encoding) |
| Vocabulary size | 16,000 |
| Character coverage | 1.0 (100%) |
| `pad_id` | 0 |
| `unk_id` | 1 |
| `bos_id` | 2 (beginning of sequence) |
| `eos_id` | 3 (end of sequence) |
| Input sentence limit | 2,000,000 sentences |
| Multi-threaded | Yes (`os.cpu_count()`) |
| Large corpus mode | `train_extremely_large_corpus=True` |

### Tokenizer Training Steps

```bash
python tokenizer/build_tokenizer.py [--vocab_size 16000] [--input story.txt]
```

1. **Preprocessing:** Stream `story.txt`, clean Wikipedia artifacts (`@-@`, `@.@`, `@,@`),
   lowercase all text.
2. **Training:** SentencePiece BPE training with streaming (4 MB chunks).
3. **Verification:** Load and verify the trained model.

**Output files:** `tokenizer/spm.model`, `tokenizer/spm.vocab`

### Corpus Encoding

```bash
python tokenizer/encode_corpus.py [--val_fraction 0.05]
```

Encodes the entire corpus into binary `.bin` files compatible with the mmap-based dataset:

| File | Content | Split |
|------|---------|-------|
| `data/train.bin` | uint16 token ID sequence | 95% of corpus |
| `data/val.bin` | uint16 token ID sequence | 5% of corpus |

The mmap format means the training loop doesn't load the full dataset into RAM — it
reads random windows from disk.

---

## Text Preprocessing Pipeline

**File:** `text_utils.py`

All text goes through a cleaning pipeline before being used for training or inference:

### Cleaning Steps

1. **NFC Unicode normalization** (`unicodedata.normalize("NFC", text)`)
2. **Lowercase** (configurable via `lowercase=True`)
3. **Whitespace collapse** (`re.sub(r"\s+", " ", text).strip()`)

### Wikipedia Artifact Removal

The corpus (`story.txt`) is from WikiText and contains special patterns:

| Pattern | Replacement | Example |
|---------|-------------|---------|
| `@ - @` | `-` | `1990 @ - @ 91` → `1990-91` |
| `@ . @` | `.` | `J @ . @ K` → `J.K` |
| `@ , @` | `,` | — |
| Stray `@` | removed | — |

### Two Tokenization Modes

| Mode | Used By | How Punctuation Is Handled |
|------|---------|---------------------------|
| **TR Token mode** | N-gram engine | Punctuation replaced with `TR001`..`TR017` placeholder tokens |
| **Pure BPE mode** | Transformer engine | SentencePiece handles punctuation natively as BPE pieces |

The N-gram model's `_prepare_for_lookup()` method applies: `full_clean → replace_punctuation_with_tokens → split`.
The Transformer's `encode()` function applies: `clean_wikipedia_artifacts → full_clean → sp.encode`.

---

## Web UI

**Files:** `templates/index.html`, `static/js/main.js`, `static/css/style.css`

### Editor & Keyboard Shortcuts

The editor is a two-layer design:

- **User layer:** A `<textarea>` where the user types (z-index: 2, transparent background)
- **Ghost layer:** A `<div>` positioned behind the textarea (z-index: 1) that shows gray
  suggestion text. The "typed" portion is rendered transparent so only the suggestion is
  visible.

| Key Combination | Action | API Endpoint | Engine |
|----------------|--------|-------------|--------|
| **Tab** | Accept one word from the ghost suggestion | — (client-side) | N-gram only |
| **Shift + Tab** | Generate a complete sentence | `/predict_sentence` | N-gram or Transformer |
| **Ctrl + Shift + Tab** | Generate a full paragraph (5 sentences) | `/predict_paragraph` | N-gram or Transformer |
| **Escape** | Dismiss the ghost suggestion | — (client-side) | — |

**Tab (word acceptance) logic:**
- The suggestion text (after the user's typed text) is parsed.
- Everything up to the first inter-word space is accepted (one word at a time).
- If only one token remains, accept everything.
- After acceptance, a fresh ghost text is immediately fetched from `/predict_next`.

**Debouncing:** Input events are debounced at 100ms to avoid flooding the server. Ghost
text is cleared immediately on keystroke so stale suggestions disappear instantly.

**Scroll sync:** The ghost layer's scroll position is synchronized with the textarea so
the suggestion stays aligned during scrolling.

### Engine Selector & Temperature Slider

**Engine dropdown (`#engine-select`):**
- `N-gram` — default, shows backoff order and status
- `Transformer` — disabled if not loaded

When `Transformer` is selected, ghost text (Tab) is disabled — only Shift+Tab and
Ctrl+Shift+Tab work. Ghost text is N-gram only because it needs to be deterministic
and fast (keystroke-level latency).

**Temperature slider (`#temperature-slider`):**
- Range: 0.5 to 2.5
- Step: 0.1
- Default: 1.0
- Display: Current value shown as text next to the slider
- Sent as query parameter `temperature=` to all generation endpoints

### N-gram Probability Visualisation Panel

Below the editor, **seven probability cards** display real-time N-gram statistics that
update as the user types:

| Card | Badge Color | Context | What It Shows |
|------|-------------|---------|---------------|
| **1-GRAM** | Green (#34a853) | — (overall frequency) | Top 10 most frequent words in corpus |
| **2-GRAM** | Pink (#f5576c) | Last 1 word | Top 5 next words after the current word |
| **3-GRAM** | Blue (#00b0ff) | Last 2 words | Top 5 next words after the last 2-word context |
| **4-GRAM** | Orange (#ff6b35) | Last 3 words | Top 5 next words after the last 3-word context |
| **5-GRAM** | Purple (#7c3aed) | Last 4 words | Top 5 next words after the last 4-word context |
| **6-GRAM** | Teal (#0d9488) | Last 5 words | Top 5 next words after the last 5-word context |
| **7-GRAM** | Rose (#e11d48) | Last 6 words | Top 5 next words after the last 6-word context |

Each card shows:
- A colored badge with the gram order
- The context words (shown in the header)
- A ranked list of candidates with word text, probability percentage, and a visual bar

The higher-order cards go **increasingly sparse** — this is the visual demonstration of
why Stupid Backoff exists: by the time you reach 7-gram, most contexts have zero or one
observed continuation.

**Info banner** at the bottom explains Temperature and the backoff order.

---

## Flask API Endpoints

**File:** `app.py`

All endpoints return JSON. The server runs on `http://localhost:5000`.

### `GET /`

Renders the web UI (`index.html`). Passes template variables:
- `ngram_ready` (bool), `ngram_error` (str), `ngram_max_n` (int)
- `transformer_ready` (bool), `transformer_error` (str)

### `GET /predict_next`

Returns the full ghost-layer text for inline word suggestion. Called on every keystroke
(debounced 100ms). Always uses the N-gram engine.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | `""` | Current editor content |
| `temperature` | float | `1.0` | Sampling temperature |

**Response:**
```json
{ "ghost": "the cat sat on the mat." }
```

### `GET /predict_sentence`

Generates text until the first sentence-ending punctuation (or max tokens).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | `""` | Current editor content (prompt) |
| `engine` | string | `"ngram"` | `"ngram"` or `"transformer"` |
| `max_tokens` | int | `100` | Maximum new tokens/words |
| `temperature` | float | `1.0` | Sampling temperature (N-gram only) |

**Response:**
```json
{ "completion": " sat on the mat." }
```

**Error response (503):**
```json
{
  "error": "Transformer not loaded.",
  "hint": "Train the model on Colab and place checkpoints/transformer.pt here."
}
```

### `GET /predict_paragraph`

Generates a multi-sentence paragraph.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | `""` | Prompt text |
| `engine` | string | `"ngram"` | `"ngram"` or `"transformer"` |
| `max_sentences` | int | `5` | Number of sentences to generate |
| `temperature` | float | `1.0` | Sampling temperature (N-gram only) |

**Response:**
```json
{ "completion": " sat on the mat. the dog barked loudly. ..." }
```

### `GET /probabilities`

Returns unigram through 7-gram probability tables for the UI panels.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | `""` | Current editor content |
| `use_tokens` | string | `"true"` | Whether to use TR token mode |

**Response:**
```json
{
  "unigram":      [{"word": "the", "probability": 6.2}, ...],
  "bigram":       [{"word": "cat", "probability": 12.5}, ...],
  "trigram":      [{"word": "sat", "probability": 25.0}, ...],
  "fourgram":     [{"word": "on",  "probability": 40.0}, ...],
  "fivegram":     [...],
  "sixgram":      [...],
  "sevengram":    [...],
  "current_word": "cat",
  "context":      "the cat",
  "context4":     "on the cat",
  "context5":     "sat on the cat",
  "context6":     "he sat on the cat",
  "context7":     "once he sat on the cat"
}
```

Each probability array contains up to 5 items (or 10 for unigram), each with:
- `word` (string): the predicted next word (with TR tokens restored to real punctuation)
- `probability` (float): percentage (0-100)

### `GET /health`

Returns the status of both engines.

**Response:**
```json
{
  "ngram": {
    "loaded": true,
    "max_n": 7,
    "vocab": 62531,
    "error": null
  },
  "transformer": {
    "loaded": false,
    "checkpoint": "checkpoints/transformer.pt",
    "error": "Checkpoint not found (checkpoints/transformer.pt)..."
  }
}
```

---

## Project Structure

```
.
├── app.py                              # Flask API server — all HTTP endpoints
├── model.py                            # NgramPredictor (Stupid Backoff, orders 1-7)
├── text_utils.py                       # Text cleaning, TR token system, SentencePiece wrappers
├── create_pickle.py                    # Builds saved_model.pkl from story.txt
├── story.txt                           # Wikipedia corpus (~514 MB)
├── training_sample.txt                 # Small text sample for quick testing
├── saved_model.pkl                     # Trained N-gram model (generated by create_pickle.py)
├── requirements.txt                    # Python dependencies
│
├── transformer/                        # Decoder-only GPT (alternative engine)
│   ├── __init__.py
│   ├── model.py                        # GPT architecture (GPTConfig, CausalSelfAttention,
│   │                                   #   FeedForward, Block, GPT)
│   ├── sample.py                       # TransformerEngine — generation with KV-cache
│   │                                   #   (SamplingConfig, generate, generate_until_sentence_end,
│   │                                   #    generate_paragraph)
│   ├── train.py                        # Training loop (AdamW, cosine LR, AMP, gradient clipping)
│   └── dataset.py                      # BinDataset — mmap-based infinite-stream dataset
│
├── tokenizer/                          # SentencePiece BPE tokenizer (vocab = 16,000)
│   ├── __init__.py
│   ├── spm.model                       # Trained BPE tokenizer model
│   ├── build_tokenizer.py              # Train the tokenizer from story.txt
│   └── encode_corpus.py                # Encode corpus to uint16 .bin files for training
│
├── notebooks/
│   ├── colab_train.ipynb               # Colab notebook — trains the Transformer on GPU
│   └── trainexample_text.py            # Training example script
│
├── checkpoints/
│   └── transformer.pt                  # Trained Transformer checkpoint
│
├── static/
│   ├── css/
│   │   └── style.css                   # Full UI styling (CSS variables, responsive grid,
│   │                                   #   ghost layer, probability cards, temperature slider)
│   └── js/
│       └── main.js                     # Client-side logic (ghost rendering, keyboard shortcuts,
│                                       #   debounced API calls, probability panel updates)
│
├── templates/
│   └── index.html                      # Jinja2 template (editor, engine selector, 7 probability
│                                       #   cards, temperature slider, keyboard hints)
│
├── extra_files/
│   └── sonuc_belgeleri/                # Result documents
│
├── GUIDE.md                            # Project guide
├── INIT.md                             # Initialization notes
└── USAGE.md                            # Usage documentation
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- pip
- (Optional) Google Colab or Kaggle account for Transformer training

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| Flask | 3.0.0 | Web framework, HTTP API |
| gunicorn | 21.2.0 | Production WSGI server |
| torch | 2.1.0 | Transformer model (PyTorch) |
| tqdm | 4.66.1 | Progress bars (corpus encoding) |
| sentencepiece | 0.2.0 | BPE tokenizer (training + inference) |
| numpy | 1.26.4 | Array operations (mmap dataset) |

### 2. Build the N-gram model

```bash
python create_pickle.py
```

This reads the first 10 million characters of `story.txt`, tokenizes with TR tokens,
and builds n-gram tables for orders 1-7.

- **Input:** `story.txt` (Wikipedia corpus, ~514 MB)
- **Output:** `saved_model.pkl` (~100-400 MB depending on data size)
- **Time:** 2-5 minutes on a normal laptop
- **Intermediate file:** `story_tokenized_temp.txt` (cleaned, TR-tokenized text)

### 3. Start the server

```bash
python app.py
```

Open `http://localhost:5000` in a browser.

**Startup output:**
```
============================================================
  AI Text Generator
============================================================
  N-gram     : OK  (max_n=7, vocab=62,531)
  Transformer: not loaded  (Checkpoint not found...)

  http://localhost:5000
============================================================
```

---

## Training Pipeline

### N-gram Model Build

**File:** `create_pickle.py`

```bash
python create_pickle.py
```

**Pipeline:**
1. Read first `READ_LIMIT` (10M) characters from `story.txt`
2. Preprocess: clean Wikipedia artifacts → lowercase → replace punctuation with TR tokens
3. Write tokenized text to `story_tokenized_temp.txt`
4. Train NgramPredictor: build unigram, bigram, trigram, 4-gram through 7-gram tables
5. Serialize to `saved_model.pkl` as a Python dict:
   ```python
   {
       "word_counts":    Counter,
       "unigram_counts": Counter,
       "bigram_counts":  {word: {next_word: count, ...}, ...},
       "trigram_counts": {(w1, w2): {w3: count, ...}, ...},
       "ngram_counts":   {4: {...}, 5: {...}, 6: {...}, 7: {...}},
       "max_n":          7,
       "total_words":    int,
   }
   ```

### Tokenizer Training

**File:** `tokenizer/build_tokenizer.py`

```bash
python tokenizer/build_tokenizer.py [--vocab_size 16000] [--input story.txt]
```

Trains a SentencePiece BPE tokenizer on the full corpus. Uses streaming (4 MB chunks) to
handle the large file without loading it all into memory.

### Corpus Encoding

**File:** `tokenizer/encode_corpus.py`

```bash
python tokenizer/encode_corpus.py [--val_fraction 0.05]
```

Encodes the entire corpus into uint16 binary files using the trained SentencePiece model.
Splits 95% train / 5% validation.

### Transformer Training (Colab/Kaggle)

**File:** `transformer/train.py`, `notebooks/colab_train.ipynb`

```bash
python transformer/train.py \
    --data_dir data \
    --tokenizer tokenizer/spm.model \
    --steps 20000 \
    --batch_size 32 \
    --block_size 256
```

**Training configuration:**

| Parameter | Default | CLI Flag | Description |
|-----------|---------|----------|-------------|
| Data directory | `data` | `--data_dir` | Contains train.bin and val.bin |
| Tokenizer | `tokenizer/spm.model` | `--tokenizer` | SentencePiece model path |
| Output directory | `checkpoints` | `--out_dir` | Where checkpoints are saved |
| Total steps | 20,000 | `--steps` | Training iterations |
| Warmup steps | 500 | `--warmup_steps` | Linear warmup for learning rate |
| Batch size | 32 | `--batch_size` | Sequences per training step |
| Block size | 256 | `--block_size` | Context window length (tokens) |
| Layers | 6 | `--n_layer` | Transformer blocks |
| Heads | 6 | `--n_head` | Attention heads |
| Model dim | 384 | `--d_model` | Embedding dimension |
| FFN dim | 1536 | `--d_ff` | Feed-forward inner dimension |
| Dropout | 0.1 | `--dropout` | Dropout rate |
| Learning rate | 3e-4 | `--lr` | Peak learning rate |
| Weight decay | 0.1 | `--weight_decay` | AdamW weight decay |
| Gradient clip | 1.0 | `--grad_clip` | Max gradient norm |
| Eval interval | 500 | `--eval_interval` | Steps between validation evaluations |
| Eval iterations | 50 | `--eval_iters` | Batches per evaluation |
| Save interval | 2000 | `--save_interval` | Steps between periodic checkpoints |
| Resume | None | `--resume` | Path to checkpoint to resume from |

**Optimizer:** AdamW with betas=(0.9, 0.95)

**Learning rate schedule:** Linear warmup → Cosine annealing (min ratio = 0.1)

```
LR
 │    ╱‾‾‾╲
 │   ╱      ╲
 │  ╱        ╲
 │ ╱          ╲_______________
 │╱
 └────────────────────────────→ steps
   warmup    cosine decay
   (500)     (500 → 20000)
```

**Mixed precision:** Automatic Mixed Precision (AMP) with float16 on CUDA, GradScaler.

**Checkpoints saved:**
- `checkpoints/transformer.pt` — best validation loss (overwritten when improved)
- `checkpoints/transformer_step_{N}.pt` — periodic snapshots every 2000 steps

**Checkpoint contents:**
```python
{
    "model":     state_dict,
    "optim":     optimizer_state,
    "scheduler": scheduler_state,
    "config":    GPTConfig.__dict__,
    "step":      int,
    "best_val":  float,
}
```

**Colab workflow:**
1. Open `notebooks/colab_train.ipynb` in Google Colab
2. Mount Google Drive and set `DRIVE_ROOT`
3. Run all cells — trains on T4 GPU
4. Download `checkpoints/transformer.pt` to local repo
5. Restart `app.py` — Transformer option becomes available

---

## Configuration Reference

### Environment Variables (`app.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `TRANSFORMER_CKPT` | `checkpoints/transformer.pt` | Path to Transformer checkpoint |
| `TOKENIZER_PATH` | `tokenizer/spm.model` | Path to SentencePiece model |

### N-gram Build Settings (`create_pickle.py`)

| Variable | Value | Description |
|----------|-------|-------------|
| `MAX_N` | 7 | Highest n-gram order |
| `READ_LIMIT` | 10,000,000 | Characters to read from story.txt (0 = full file) |
| `INPUT_FILE` | `story.txt` | Input corpus file |
| `TEMP_FILE` | `story_tokenized_temp.txt` | Intermediate tokenized file |
| `OUTPUT_FILE` | `saved_model.pkl` | Output model file |

### Transformer Config (`transformer/model.py`)

```python
@dataclass
class GPTConfig:
    vocab_size: int   = 16000   # BPE vocabulary size
    block_size: int   = 256     # Maximum context window
    n_layer:    int   = 6       # Transformer blocks
    n_head:     int   = 6       # Attention heads
    d_model:    int   = 384     # Embedding / model dimension
    d_ff:       int   = 1536    # FFN inner dimension
    dropout:    float = 0.1     # Dropout rate
    pad_id:     int   = 0       # Padding token ID
```

### Sampling Config (`transformer/sample.py`)

```python
@dataclass
class SamplingConfig:
    max_new_tokens:     int   = 80     # Max tokens to generate
    temperature:        float = 0.85   # Softmax temperature
    top_k:              int   = 40     # Top-k filtering
    top_p:              float = 0.92   # Nucleus sampling threshold
    repetition_penalty: float = 1.15   # Repetition penalty (>1 = penalize)
```

### CSS Design Tokens (`static/css/style.css`)

| Variable | Value | Usage |
|----------|-------|-------|
| `--bg-primary` | `#ffffff` | Main background |
| `--bg-secondary` | `#f8f9fa` | Card/editor background |
| `--text-primary` | `#1a1d23` | Main text color |
| `--text-ghost` | `#999999` | Ghost suggestion text |
| `--accent-primary` | `#4285f4` | Google Blue — links, highlights |
| `--unigram-color` | `#34a853` | 1-gram card badge |
| `--bigram-color` | `#f5576c` | 2-gram card badge |
| `--trigram-color` | `#00b0ff` | 3-gram card badge |
| `--fourgram-color` | `#ff6b35` | 4-gram card badge |
| `--fivegram-color` | `#7c3aed` | 5-gram card badge |
| `--sixgram-color` | `#0d9488` | 6-gram card badge |
| `--sevengram-color` | `#e11d48` | 7-gram card badge |

---

## Developer

[@alperenkarakaya](https://github.com/alperenkarakaya)
