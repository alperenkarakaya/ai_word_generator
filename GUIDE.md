# AI Text Generator — Complete Technical Guide

## Table of Contents
1. [What this project is](#1-what-this-project-is)
2. [Why N-gram fails and why Transformers work](#2-why-n-gram-fails-and-why-transformers-work)
3. [Tokenization — how text becomes numbers](#3-tokenization--how-text-becomes-numbers)
4. [The Transformer architecture — every component explained](#4-the-transformer-architecture--every-component-explained)
5. [Training — how the model learns](#5-training--how-the-model-learns)
6. [Generation — how text is produced](#6-generation--how-text-is-produced)
7. [Project file structure](#7-project-file-structure)
8. [Step-by-step: training on Google Colab](#8-step-by-step-training-on-google-colab)
9. [Running the app locally](#9-running-the-app-locally)
10. [Using the web interface](#10-using-the-web-interface)
11. [Roadmap: paragraphs, content, Q&A](#11-roadmap-paragraphs-content-qa)

---

## 1. What this project is

A text generation system that completes sentences and paragraphs from a typed prompt. The project traces the historical arc of language modelling:

```
Statistics (N-gram)  ──►  Attention (Transformer)  ──►  Long context  ──►  Q&A
      done                       ← YOU ARE HERE →
```

The training corpus is the **English Wikipedia text** (`story.txt`, ~514 MB, WikiText-103 format).

The user interface is a **Flask web app** with a live text editor. The N-gram model still powers the three probability-display panels in the UI — showing real-time statistics as an educational comparison with the Transformer.

---

## 2. Why N-gram fails and why Transformers work

### The N-gram approach

An N-gram model counts how often word B follows word A (bigram), or how often word C follows the pair (A, B) (trigram). To generate the next word, it looks at the last 1, 2, or 3 words and samples from the observed frequency distribution.

**The fundamental problem** is that its memory window is fixed at N−1 words. A trigram remembers 2 words. After generating the 4th word, it has completely forgotten the first word. It cannot know what the sentence is _about_.

Consider generating after the prompt `"The capital of France"`:
- A trigram sees only `"of France"`.
- It has no idea that `"The capital"` appeared before.
- It samples any word that commonly follows `"of France"` — which could be `"and"`, `"the"`, `"is"`, or anything.

There is also a second problem: **weighted random sampling with no filtering**. The N-gram samples from the _full_ distribution, which includes very low-probability words. Even if "Paris" has a 40% chance, there is a 60% chance of picking something else. Over a 15-word sentence, the odds of choosing a sensible word every time drop exponentially.

### Why attention solves both problems

A Transformer reads the _entire_ prompt simultaneously. Every token in the sequence can directly look at every other token through the **attention mechanism**. When generating word 40, the model has direct access to word 1. There is no forgetting.

Additionally, the model is not a frequency table — it is a trained neural network that has internalized English grammar and factual patterns from millions of Wikipedia sentences. It does not merely pick from a list of observed followers; it _reasons_ (implicitly) about what should come next given everything it has read.

---

## 3. Tokenization — how text becomes numbers

Neural networks cannot process strings. Every piece of text must be converted to a sequence of integers before the model ever sees it.

### Step 1 — Punctuation tokens (TR001 … TR017)

Before any machine-learning tokenization, punctuation marks are replaced with explicit named tokens defined in `text_utils.py`:

```
'.'  →  TR001      ','  →  TR002      '!'  →  TR003
'?'  →  TR004      ';'  →  TR005      ':'  →  TR006
'-'  →  TR007      '('  →  TR010      ')'  →  TR011
'"'  →  TR012      "'"  →  TR013      '…'  →  TR016
'\n' →  TR017
```

The sentence `"Hello, world."` becomes `"Hello TR002 world TR001"`.

These tokens are not arbitrary. They carry structural meaning:
- TR001, TR003, TR004 are **sentence-enders**. The generation loop checks for these and stops.
- TR017 is a **paragraph break**. Phase 3 will use it to stop paragraph generation.
- Having punctuation as explicit tokens means the model learns _where sentences end_ as a learnable pattern, not as a hard-coded rule.

### Step 2 — SentencePiece BPE (Byte Pair Encoding)

After punctuation substitution, the remaining text is tokenized with a **learned subword algorithm** called Byte Pair Encoding.

**How BPE works:**
1. Start with individual characters as the vocabulary: `['a', 'b', 'c', ...]`.
2. Count every adjacent pair in the training corpus.
3. Merge the most frequent pair into a new symbol. Repeat.
4. After enough merges you reach the target vocabulary size (16,000 here).

The result is a vocabulary of character sequences at various levels of granularity. Common words get their own token; rare words are split into familiar sub-pieces:

```
"the"        → one token   (very common)
"history"    → "hist" + "ory"  (common subwords)
"discovered" → "dis" + "cover" + "ed"
"Valkyria"   → "Val" + "ky" + "ria"   (rare name, split into pieces)
```

**Why this matters:** English morphology is rich — adding `-ing`, `-ed`, `-tion`, `-ness` creates thousands of word forms. If every form were a separate token we would need a vocabulary of millions. BPE handles all of these through shared sub-pieces, keeping the vocabulary small (16k) while still representing every word in the corpus.

**The critical constraint — TR tokens must be single IDs:**
The TR001…TR017 tokens are declared as `user_defined_symbols` when training SentencePiece. This forces BPE to never split them. `TR001` must always encode to _one_ integer, not `['TR', '001']` or `['T', 'R', '001']`. The generation loop stops on the sentence-end integer IDs. If a TR token were split, the loop would never see the stop signal and would generate forever.

`tokenizer/build_tokenizer.py` ends with an assertion that checks this:
```python
for tok in ALL_TR_TOKENS:
    ids = sp.encode(tok, out_type=int)
    assert len(ids) == 1, f"{tok} split into {ids}"
```

### The binary corpus files

After the tokenizer is trained, `tokenizer/encode_corpus.py` converts the entire 514 MB `story.txt` into a flat array of uint16 integers written to `data/train.bin` (95%) and `data/val.bin` (5%).

`uint16` stores values 0–65535. Since our vocabulary has 16,000 tokens, each id fits in 16 bits (2 bytes). The 514 MB text compresses to roughly 200–350 MB of binary data.

During training, the dataset is loaded with `numpy.memmap`, which maps the file into virtual memory without loading it all into RAM at once. When the training code requests a batch, only the needed bytes are read from disk. This is how a T4 Colab GPU (16 GB RAM) can train on a 350 MB file that would otherwise overflow its memory budget when stored as Python objects.

---

## 4. The Transformer architecture — every component explained

The model is defined in `transformer/model.py`. It is a **decoder-only Transformer** — the same fundamental design as GPT. Here is a top-down walkthrough.

### Big picture

```
                         (repeated N_LAYER times)
                        ┌─────────────────────────┐
token ids               │  LayerNorm               │
    │                   │       ↓                  │
    ▼                   │  Causal Self-Attention    │
Token Embedding         │       ↓                  │
    +                   │  Residual Add             │
Positional Embedding    │       ↓                  │
    │                   │  LayerNorm               │
    ▼                   │       ↓                  │
   x ─────────────────► │  Feed-Forward Network    │ ──► x_out
                        │       ↓                  │
                        │  Residual Add            │
                        └─────────────────────────┘
                                  │
                              LayerNorm
                                  │
                              LM Head  (Linear, weight-tied to Token Embedding)
                                  │
                           Logits over vocabulary
```

### Token embedding

Each integer token ID is mapped to a dense vector of size `d_model` (384).
This is a lookup table: `nn.Embedding(vocab_size, d_model)`.

Conceptually, nearby vectors in this space represent tokens with similar meanings. After training, words like "king" and "queen", or "ran" and "run", end up close together in the 384-dimensional space.

### Positional embedding

Attention has no built-in notion of order — it sees all tokens simultaneously. Without positional information, the model would treat "dog bites man" and "man bites dog" identically.

This model uses **learned positional embeddings**: another lookup table `nn.Embedding(block_size, d_model)`, where position 0 has one learnable vector, position 1 has another, and so on up to `block_size − 1` (255). The positional vector is added to the token embedding before the first block.

The final input to the transformer stack at position `i` is therefore:

```
x[i] = token_embedding[token_id[i]] + position_embedding[i]
```

### Transformer block (repeated 6 times)

Each of the 6 layers applies the same two sub-operations, each with a residual connection:

```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

The **residual connection** (`x + ...`) means the output of each sub-layer is _added_ to its input. This gives gradients a direct path back to the earliest layers during backpropagation, making it possible to train 6 (or 96, in GPT-4) layers without vanishing gradients.

**Pre-norm** means `LayerNorm` is applied _before_ the sub-layer, not after. Empirically this leads to more stable training.

### Causal multi-head self-attention

This is the heart of the Transformer. Every token in the sequence attends to every other token — but only tokens _before_ it (causal, aka "masked" attention), because during generation the model cannot see future tokens.

**Step 1 — Project to Q, K, V**

The input `x` (shape: `[batch, seq_len, d_model]`) is linearly projected into three matrices:
- **Q (Queries)**: "What am I looking for?"
- **K (Keys)**: "What do I contain?"
- **V (Values)**: "What do I pass on if someone attends to me?"

All three have shape `[batch, seq_len, d_model]`.

**Step 2 — Split into heads**

The `d_model=384` dimension is split into `n_head=6` heads of `head_dim=64` each. Each head independently performs attention over a different learned subspace. This lets the model attend to multiple different types of relationships simultaneously (e.g., one head for grammar, another for coreference, another for proximity).

**Step 3 — Scaled dot-product attention**

For each head:
```
Attention(Q, K, V) = softmax( Q · Kᵀ / √head_dim ) · V
```

`Q · Kᵀ` produces a `[seq_len, seq_len]` matrix of raw attention scores. Each entry `[i, j]` measures how much token `i` should attend to token `j`.

Dividing by `√head_dim` (√64 = 8) prevents the dot products from growing too large, which would push the softmax into saturation and make gradients vanish.

The **causal mask** is applied before the softmax: all positions `j > i` are set to `-∞`. After softmax, these become 0, so token `i` receives no information from future tokens.

Softmax turns the scores into a probability distribution summing to 1. The output is a weighted sum of the Value vectors.

**Step 4 — Concatenate heads, project back**

The outputs from all 6 heads are concatenated and projected back to `d_model` with a linear layer.

**In plain English:** each token computes a query ("I'm interested in what relates to my meaning"). Every other token (in the past) responds with a key ("here is what I contain"). The dot product measures alignment. High alignment → the current token receives a lot of that past token's value. Low alignment → ignored. This is a soft, differentiable _memory lookup_ over the entire context.

### Feed-forward network (FFN)

After attention, each token's representation passes through a small two-layer MLP applied independently at every position:

```
FFN(x) = Linear(GELU(Linear(x)))
```

The hidden layer is `d_ff = 1536 = 4 × d_model`. The GELU (Gaussian Error Linear Unit) activation is a smooth approximation to ReLU, empirically better for language models.

The FFN acts as the model's "memory" for factual associations. Attention selects _which_ past tokens are relevant; the FFN transforms the combined representation into a semantically richer one. Research has shown that factual knowledge (e.g., "Paris is the capital of France") is largely stored in FFN weights.

### Final LayerNorm and LM Head

After the last block, one more LayerNorm is applied, then a linear projection (`d_model → vocab_size`) produces raw scores (logits) for every token in the vocabulary.

**Weight tying:** the LM Head linear layer shares its weight matrix with the Token Embedding. This halves the parameter count for these large matrices (`vocab_size × d_model = 16000 × 384 ≈ 6M params`) and is a well-established trick that also improves generalization.

### Parameter count

With `n_layer=6, d_model=384, n_head=6, d_ff=1536, vocab_size=16000, block_size=256`:

| Component | Parameters |
|---|---|
| Token + Position Embedding | 16000 × 384 + 256 × 384 ≈ 6.2M |
| Attention (QKV + Proj) × 6 | 4 × 384² × 6 ≈ 3.5M |
| FFN × 6 | 2 × 384 × 1536 × 6 ≈ 7.1M |
| LayerNorms | negligible |
| LM Head | shared with embedding, 0 extra |
| **Total** | **~30M** |

For comparison: GPT-2 small has 117M, GPT-3 has 175B. Our model is tiny but sufficient to produce coherent English sentences after training on 514 MB of Wikipedia.

---

## 5. Training — how the model learns

Training is implemented in `transformer/train.py`.

### The task: next-token prediction

The model is trained on one deceptively simple task: given a sequence of tokens, predict what comes next at every position.

From a single training sample of 256 tokens `[t₀, t₁, ..., t₂₅₅]`, we get 255 training examples simultaneously:
- Input `[t₀]` → target `t₁`
- Input `[t₀, t₁]` → target `t₂`
- Input `[t₀, t₁, t₂]` → target `t₃`
- ... and so on.

This is why training is efficient: one forward pass through one sequence yields 255 gradient signals.

### Loss function: cross-entropy

At each position `i`, the model produces a probability distribution over the 16,000 vocabulary tokens. The **cross-entropy loss** measures how much probability the model assigned to the _correct_ token:

```
loss_i = -log( P(correct token at position i) )
```

If the model is completely certain and correct, `P = 1.0`, `loss = 0`.
If the model is completely wrong, `P ≈ 0`, `loss → ∞`.
For a random model (uniform over 16k tokens): `loss = ln(16000) ≈ 9.68`.

The total loss is the average across all positions in the batch.

### Reference scale

| Situation | Loss (nats) |
|---|---|
| Random model, 16k vocab | 9.68 |
| After 1k steps | ~6.5 |
| After 5k steps | ~5.2 |
| After 20k steps (target) | < 4.5 |
| Human-level English | ~1.5–2.0 |

We will not reach human-level with 30M parameters and a few hours of training. But dropping below ~4.5 is where output becomes recognizably grammatical and topically coherent.

### AdamW optimiser

Gradients from backpropagation tell us which direction to adjust every parameter. **Adam** (Adaptive Moment Estimation) maintains a running average of past gradients (momentum) and past squared gradients (adaptive learning rate per parameter). This makes it converge much faster than plain SGD on sparse or noisy problems like language.

**W** in AdamW stands for **weight decay** — a regularisation term that penalises large weights, added directly to the weights rather than through the gradient (the correct formulation; standard L2 regularisation interacts badly with Adam).

Settings used:
- `lr = 3e-4` (peak learning rate)
- `betas = (0.9, 0.95)` — momentum memory. The 0.95 (vs. the usual 0.999) gives the optimiser a shorter memory, which works better for transformers.
- `weight_decay = 0.1`

### Cosine learning rate schedule with warmup

Training with a fixed learning rate often leads to instability at the start (the randomly initialised model produces large erratic gradients) and slow convergence at the end.

The schedule has two phases:

```
steps 0 → 500 (warmup):
    LR rises linearly from 0 → 3e-4.
    Reason: random init → large gradients → a small LR prevents blowing up weights.

steps 500 → 20000 (cosine decay):
    LR follows a cosine curve from 3e-4 down to 3e-5 (10% of peak).
    Reason: cosine is smooth, avoids abrupt LR drops that can destabilise training.
```

### Mixed precision (AMP)

On a CUDA GPU, model weights are stored in float32 (4 bytes/number). With **Automatic Mixed Precision**, the forward pass and gradient computation run in float16 (2 bytes/number), which is roughly twice as fast on modern GPU tensor cores. A `GradScaler` automatically scales the loss to prevent float16 underflow during backward.

The master weights remain in float32, so numerical precision for weight updates is not compromised.

### Gradient clipping

After backprop but before the optimiser step, all gradients are rescaled so their global L2 norm does not exceed 1.0:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

This prevents the "exploding gradient" problem, where a single bad batch sends a parameter flying to ±infinity.

### Checkpointing

Every 2000 steps, the training script saves a checkpoint:
```
checkpoints/transformer_step_2000.pt
checkpoints/transformer_step_4000.pt
...
checkpoints/transformer.pt   ← always the best val-loss checkpoint
```

Each checkpoint contains the full model state (`model_state_dict`), optimiser state, scheduler state, current step, and the `GPTConfig` dictionary. This last item is critical: it lets you load the checkpoint without knowing in advance what architecture parameters were used.

### Train vs validation loss

Every 500 steps, the script evaluates on `data/val.bin` (the 5% held-out set). Comparing train and val loss tells you if the model is memorising the training data (overfitting) or learning general patterns (generalising):

- `val_loss ≈ train_loss` → healthy generalisation.
- `val_loss >> train_loss` → overfitting. Increase dropout or reduce model size.
- `val_loss < train_loss` → unusual; suggests the val set is easier than train.

---

## 6. Generation — how text is produced

Generation is implemented in `transformer/sample.py` in the `TransformerEngine` class.

### Autoregressive decoding

The model generates text one token at a time. At each step:
1. Feed the current context (prompt + everything generated so far) → get logits over vocab.
2. Sample one token from the logits.
3. Append that token to the context.
4. Repeat.

This is called **autoregressive** generation: each new token becomes part of the input for the next prediction.

### KV-cache: why we need it and how it works

On step 1, the full context (say 20 tokens) is processed. On step 2, we have 21 tokens. On step 3, 22. If we recomputed the full forward pass every step, the cost would be O(step²) — quadratic in sequence length.

The KV-cache solves this. During the attention computation, every token computes a **Key** and **Value** matrix. These do not change for previously-seen tokens (because their inputs did not change). So we **cache** (save) the K and V tensors after the first forward pass. On each subsequent step, we only feed the one new token:

```
Step 1 (full prompt, 20 tokens):   model processes 20 tokens → produces KV-cache (20 entries)
Step 2 (new token only):           model processes 1 token, attends to cached 20+1 → fast
Step 3 (new token only):           model processes 1 token, attends to cached 21+1 → fast
...
```

Each incremental step is O(context_length) in memory (reading the cache) but O(1) in compute relative to sequence length. This makes real-time generation in a web app practical.

**Positional offset:** when feeding a single new token at position `n`, we must tell the model where in the sequence it sits, so the positional embedding looks up the right slot. This is the `pos_offset` parameter in `GPT.forward()`. If `n_cached` tokens are in the cache, the new token sits at position `n_cached`.

**Sliding window:** the positional embedding table has `block_size` rows (0 … 255). Once the context length would reach 256, we cannot extend without the position index going out of range. At that point, we drop the oldest tokens, keeping the last 255, reset the KV-cache, and reprocess from position 0. The model loses very early context but retains recent coherence.

### Sampling strategies

After the forward pass we have a logit vector of size `vocab_size` (16,000). We must turn this into a single token. Several strategies, applied in order:

#### Temperature

```
adjusted_logits = logits / T
probs = softmax(adjusted_logits)
```

- `T = 1.0`: unchanged distribution.
- `T < 1` (e.g., 0.5): sharper distribution, model is more confident, picks likely tokens more often. Output is more predictable but can become repetitive.
- `T > 1` (e.g., 1.5): flatter distribution, model explores less-likely tokens. Output is more surprising but can become incoherent.

We use `T = 0.85` — slightly focused, but not rigid.

#### Top-k

After temperature scaling, zero out every logit except the top-k highest values. With `k = 40`, the model can only choose from the 40 most likely next tokens. This completely eliminates the long tail of improbable words that caused the N-gram's incoherence.

#### Top-p (nucleus sampling)

Sort tokens by probability (high to low). Accumulate probabilities until the running sum exceeds `p = 0.92`. Discard everything after that cutoff.

The difference from top-k: when the model is very confident (one token dominates), top-p naturally picks a small set. When the model is uncertain (many tokens equally likely), top-p allows a larger set. It adapts dynamically to the model's confidence, which top-k cannot.

#### Repetition penalty

For every token already in the context, its logit is reduced:
- If the logit is positive: `logit / penalty` (e.g., divide by 1.15 → smaller positive)
- If the logit is negative: `logit × penalty` (e.g., multiply by 1.15 → more negative)

This discourages the model from generating the same word repeatedly, a common failure mode when temperature is low.

### Sentence completion: stopping at TR001 / TR003 / TR004

`generate_until_sentence_end()` calls `_generate_ids()` with `stop_ids = {id_of_TR001, id_of_TR003, id_of_TR004}`. The generation loop checks whether the just-sampled token id is in this set and breaks immediately if so.

The sampled TR001 id is included in the output, so the generated text ends with a period, exclamation mark, or question mark.

---

## 7. Project file structure

```
uni_bitirme/
│
├── story.txt                      Training corpus, 514 MB English Wikipedia (gitignored)
│
├── app.py                         Flask web server — entry point
├── model.py                       N-gram model (powers probability panels only)
├── text_utils.py                  Shared: TR tokens, SentencePiece wrappers, cleaning
├── create_pickle.py               Builds saved_model.pkl (N-gram training)
│
├── tokenizer/
│   ├── build_tokenizer.py         Trains SentencePiece BPE on story.txt → spm.model
│   └── encode_corpus.py           story.txt → data/train.bin + data/val.bin
│
├── transformer/
│   ├── model.py                   GPT architecture (Embedding, Attention, FFN, LM Head)
│   ├── dataset.py                 Memory-mapped dataset reader (BinDataset)
│   ├── train.py                   Training loop (AdamW, cosine LR, AMP, checkpoints)
│   └── sample.py                  TransformerEngine: KV-cache generation, top-k/top-p
│
├── notebooks/
│   └── colab_train.ipynb          9-cell Colab walkthrough (Drive → train → export)
│
├── checkpoints/                   Created during training (gitignored)
│   └── transformer.pt             Best checkpoint (lowest val loss)
│
├── data/                          Created by encode_corpus.py (gitignored)
│   ├── train.bin                  95% of corpus, uint16 token ids
│   └── val.bin                    5% of corpus
│
├── tokenizer/spm.model            Created by build_tokenizer.py (gitignored)
│
├── saved_model.pkl                Pre-built N-gram model for probability panels
├── templates/index.html           Web UI
├── static/css/style.css
├── static/js/main.js
└── requirements.txt
```

**Deleted in this cleanup (Phase 2):**
- `transformer_model.py` — character-level LSTM mislabelled as a Transformer
- `train_lstm.py` — training script for the above
- `LSTM_GUIDE.md` — documentation for deleted code

---

## 8. Step-by-step: training on Google Colab

### Prerequisites

| What | Where |
|---|---|
| This repo | Pushed to your GitHub account |
| `story.txt` | Uploaded to Google Drive at `My Drive/uni_bitirme/story.txt` |
| Colab session | Runtime → Change runtime type → **T4 GPU** |

Open `notebooks/colab_train.ipynb` on Colab.

---

### Cell 0 — Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

A dialog asks for permission. After approval, your Drive is accessible at `/content/drive/MyDrive/`. This cell also creates `My Drive/uni_bitirme/` if it does not exist — this is where the final checkpoint will be saved so it survives a session timeout.

---

### Cell 1 — Clone the repo

Edit `REPO_URL` to your actual GitHub URL, then run:

```python
REPO_URL = 'https://github.com/alperenkarakaya/your-repo.git'
```

The cell clones to `/content/uni_bitirme` and changes into it with `os.chdir`. All subsequent cells run from this directory.

---

### Cell 2 — story.txt

Checks if `story.txt` exists. If not, copies it from `My Drive/uni_bitirme/story.txt`. If it is not on Drive either, the cell raises a `FileNotFoundError` with instructions.

`story.txt` is gitignored (514 MB is too large for GitHub). It must always be brought to Colab separately via Drive.

---

### Cell 3 — Install dependencies

```bash
pip install sentencepiece tqdm
```

PyTorch comes pre-installed on Colab. This cell also prints the PyTorch version and confirms CUDA is available.

---

### Cell 4 — Build tokenizer (~5–10 minutes)

```bash
python tokenizer/build_tokenizer.py --input story.txt --vocab_size 16000
```

What happens internally:
1. Reads `story.txt` in 4 MB chunks (never loads the full file).
2. For each chunk: removes Wikipedia `@-@` markers, replaces punctuation with TR tokens.
3. Writes the processed text to a temp file.
4. Calls `SentencePieceTrainer.train()` with `user_defined_symbols = TR001..TR017`.
5. Deletes the temp file.
6. Asserts every TR token encodes to exactly one id.

Expected output:
```
[1/4] Pre-processing: story.txt
      Characters written: 287,432,100
[2/4] SentencePiece BPE training (vocab_size=16000)
[3/4] Validating: tokenizer/spm.model
[4/4] Verifying all TR tokens map to single ids
  OK — 13 TR tokens verified.
Vocab size: 16000
```

If the TR token verification fails, **stop here**. The generation loop depends on these being single ids.

---

### Cell 5 — Encode corpus (~10–20 minutes)

```bash
python tokenizer/encode_corpus.py \
    --input story.txt --tokenizer tokenizer/spm.model \
    --out_dir data --val_fraction 0.05
```

Streams story.txt through the trained tokenizer, writes every token id as a uint16 to `data/train.bin` (95%) and `data/val.bin` (5%).

After this cell, run `!ls -lh data/`. You should see two files, each hundreds of MB. If either is missing or near-empty, something went wrong in the tokenizer step.

---

### Cell 6 — Train (3–6 hours on T4)

```bash
python transformer/train.py \
    --data_dir data \
    --tokenizer tokenizer/spm.model \
    --steps 20000 \
    --batch_size 32 \
    --block_size 256 \
    --eval_interval 500 \
    --save_interval 2000 \
    --warmup_steps 500 \
    --lr 3e-4
```

Every 50 steps you will see:
```
step    50 | loss 7.23 | lr 3.00e-04 | tok/s 14230
step   100 | loss 6.88 | lr 3.00e-04 | tok/s 14380
```

Every 500 steps, validation loss is reported:
```
>>> step 500 | train 5.91 | val 6.03
>>> best val. Saved: transformer.pt
```

**Loss trajectory to expect:**
- Steps 0–500: rapid drop from ~9.5 to ~6.5 (learning basic word frequencies)
- Steps 500–3000: steady drop to ~5.0–5.5 (learning grammar patterns)
- Steps 3000–20000: slower improvement toward ~4.2–4.5 (topic coherence)

**If the session expires:** re-run cells 0–1 to remount Drive and re-enter the repo. Then resume with:
```bash
python transformer/train.py ... --resume checkpoints/transformer_step_XXXX.pt
```
Use the highest-numbered step file. The optimiser state is saved so training continues seamlessly.

---

### Cell 7 — Back up to Drive

Always run this before closing the session:
```bash
cp checkpoints/transformer.pt /content/drive/MyDrive/uni_bitirme/transformer.pt
cp tokenizer/spm.model        /content/drive/MyDrive/uni_bitirme/spm.model
cp tokenizer/spm.vocab        /content/drive/MyDrive/uni_bitirme/spm.vocab
```

---

### Cell 8 — Sanity test

```python
from transformer.sample import TransformerEngine
engine = TransformerEngine('checkpoints/transformer.pt', 'tokenizer/spm.model')

for p in ['The history of', 'In the early years', 'Scientists discovered that']:
    print(engine.generate_until_sentence_end(p))
```

After good training you should see complete, grammatical sentences. They will be "Wikipedia-flavoured" (factual, encyclopedic), which is expected since that is what the model was trained on.

---

### After Colab — deploy locally

1. Download `transformer.pt` from Drive → place at `checkpoints/transformer.pt`
2. Download `spm.model` from Drive → place at `tokenizer/spm.model`
3. `pip install sentencepiece==0.2.0`
4. `python app.py`
5. Open `http://localhost:5000`

---

## 9. Running the app locally

### First-time setup

```bash
pip install -r requirements.txt
python app.py
```

**Without the transformer checkpoint**, the app starts in partial mode: probability panels work (N-gram), but sentence generation returns a 503 with an explanation.

**With the checkpoint:**
```
checkpoints/transformer.pt   ← required
tokenizer/spm.model          ← required
```

Both must exist. `python app.py` auto-detects them.

### Custom checkpoint path

```bash
TRANSFORMER_CKPT=/path/to/my_run.pt TOKENIZER_PATH=/path/to/spm.model python app.py
```

### Rebuilding the N-gram model (probability panels)

If `saved_model.pkl` is missing:
```bash
python create_pickle.py
```
This reads the first 1 MB of `story.txt` and builds the N-gram tables. Takes under a minute.

---

## 10. Using the web interface

### Editor

Type freely in the text area. The three panels below update in real time with N-gram statistics.

### Keyboard shortcuts

| Key | Action |
|---|---|
| `Shift` + `Tab` | Complete the current sentence (Transformer generates until `.`, `!`, or `?`) |
| `Ctrl` + `Shift` + `Tab` | Generate a full paragraph (Phase 3, same endpoint) |

### Probability panels

The three panels show N-gram statistics — not the Transformer's predictions. They exist to visualise what the old approach was doing, as an educational comparison:

- **1-gram**: the top 10 most frequent tokens in the corpus, ranked by overall frequency. Completely ignores what you typed.
- **2-gram**: given the last word you typed, what word most commonly follows it in Wikipedia?
- **3-gram**: given the last _two_ words, what word most commonly follows?

Compare the coherence of a 3-gram completion (which sees 2 words of context) against the Transformer completion (which sees all 256). This comparison is the academic argument of the project.

### REST API (for testing)

```bash
# Sentence completion
curl "http://localhost:5000/predict_sentence?text=The+war+began+when"

# Paragraph generation
curl "http://localhost:5000/predict_paragraph?text=Paris+is&max_sentences=4"

# N-gram probability table
curl "http://localhost:5000/probabilities?text=the+history"

# Health check
curl "http://localhost:5000/health"
```

---

## 11. Roadmap: paragraphs, content, Q&A

### Phase 3 — Paragraphs

**What changes:** continue-train the existing checkpoint with `block_size=512` so the model sees multi-sentence context during training. Add `generate_paragraph()` to `transformer/sample.py` — already stubbed in.

**Stop condition:** TR017 (newline token) OR after `max_sentences` sentence-enders.

**New endpoint:** `/predict_paragraph` — already implemented in `app.py`.

### Phase 4 — Multi-paragraph content

**What changes:** add article structure tokens to the vocabulary — `TR030 = <article_start>`, `TR031 = <title_end>`, `TR032 = <article_end>`. Re-encode the corpus preserving Wikipedia article boundaries. Continue-train at `block_size=1024`.

**New endpoint:** `/predict_content?title=...` — generates `[article_start] title [title_end] [body paragraphs] [article_end]`.

### Phase 5 — Question → Answer (optional)

**Approach: lightweight RAG (Retrieval-Augmented Generation)**

1. Split `story.txt` into ~200-token paragraph chunks.
2. Build a BM25 index (keyword search, no extra model needed).
3. For a question `q`: retrieve the top-3 most relevant chunks.
4. Build the prompt: `"Question: {q} Context: {chunk1} {chunk2} Answer:"`.
5. Call the Phase 4 generator with low temperature (0.4) and stop at the first sentence-ender.

This reuses the trained model from Phase 4 with no additional training. The model learns to "read" the context and extract an answer, much like a human reading a reference paragraph.

For better accuracy, an optional fine-tuning step can be added later: generate synthetic Q&A pairs from Wikipedia first sentences and fine-tune for a few hundred steps.
