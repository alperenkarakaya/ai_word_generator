"""
Evaluation framework for N-gram and Transformer engines.

Computes Top-1/3/5 accuracy, perplexity, bits-per-character,
latency, and memory usage for both engines on held-out test data.

Usage:
    python evaluate.py                                # full evaluation
    python evaluate.py --skip-latency --skip-memory   # quality only (~2 min)
    python evaluate.py --test-chars 100000            # more test data
"""

import argparse
import gc
import json
import math
import os
import random
import sys
import time
import tracemalloc
from collections import Counter
from datetime import datetime
from statistics import mean, median, stdev

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

STORY_PATH = "story.txt"
NGRAM_PICKLE = "saved_model.pkl"
TRANSFORMER_CKPT = os.path.join("checkpoints", "transformer.pt")
TOKENIZER_PATH = os.path.join("tokenizer", "spm.model")
NGRAM_TRAIN_LIMIT = 10_000_000
SMOOTHING_EPSILON = 1e-10


# ---------------------------------------------------------------------------
#  Progress
# ---------------------------------------------------------------------------

_t0 = time.time()


def print_progress(msg: str):
    elapsed = time.time() - _t0
    print(f"[{elapsed:6.1f}s] {msg}")


# ---------------------------------------------------------------------------
#  Test data
# ---------------------------------------------------------------------------

def prepare_test_data(story_path: str, train_limit: int, test_chars: int) -> str:
    if not os.path.exists(story_path):
        sys.exit(f"ERROR: {story_path} not found.")
    with open(story_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    if len(full_text) <= train_limit:
        sys.exit(
            f"ERROR: story.txt has only {len(full_text):,} chars, "
            f"need >{train_limit:,} for test split."
        )
    tail = full_text[train_limit:]
    dot_pos = tail.find(". ")
    if dot_pos == -1:
        dot_pos = 0
    else:
        dot_pos += 2
    raw = tail[dot_pos: dot_pos + test_chars]
    print_progress(
        f"Test data: {len(raw):,} chars from story.txt "
        f"[{train_limit + dot_pos:,} .. {train_limit + dot_pos + len(raw):,}]"
    )
    return raw


def extract_test_prompts(raw_text: str, n_prompts: int = 10,
                         words_per_prompt: int = 10) -> list:
    words = raw_text.split()
    if len(words) < n_prompts * words_per_prompt:
        n_prompts = max(1, len(words) // words_per_prompt)
    step = len(words) // n_prompts
    prompts = []
    for i in range(n_prompts):
        start = i * step
        prompt = " ".join(words[start: start + words_per_prompt])
        prompts.append(prompt)
    return prompts


# ---------------------------------------------------------------------------
#  N-gram helpers
# ---------------------------------------------------------------------------

def tokenize_for_ngram(ngram_model, text: str) -> list:
    return ngram_model._prepare_for_lookup(text)


def get_ngram_distribution(ngram_model, context: list):
    """Walk the Stupid Backoff chain and return (Counter, backoff_level)."""
    for n in range(ngram_model.max_n, 3, -1):
        if n not in ngram_model.ngram_counts:
            continue
        if len(context) < n - 1:
            continue
        key = tuple(context[-(n - 1):])
        cands = ngram_model.ngram_counts[n].get(key)
        if cands:
            return cands, n

    if len(context) >= 2:
        key = (context[-2], context[-1])
        cands = ngram_model.trigram_counts.get(key)
        if cands:
            return cands, 3

    if context:
        cands = ngram_model.bigram_counts.get(context[-1])
        if cands:
            return cands, 2

    return ngram_model.unigram_counts, 1


# ---------------------------------------------------------------------------
#  N-gram perplexity
# ---------------------------------------------------------------------------

def compute_ngram_perplexity(ngram_model, tokens: list) -> dict:
    log_prob_sum = 0.0
    n_positions = 0
    oov_count = 0
    level_counts = Counter()

    for i in range(1, len(tokens)):
        context = tokens[:i]
        target = tokens[i]
        cands, level = get_ngram_distribution(ngram_model, context)
        level_counts[level] += 1

        total = sum(cands.values())
        if total == 0:
            p = SMOOTHING_EPSILON
            oov_count += 1
        elif target in cands:
            p = cands[target] / total
        else:
            p = SMOOTHING_EPSILON
            oov_count += 1

        log_prob_sum += math.log(p)
        n_positions += 1

    if n_positions == 0:
        return {"perplexity": None, "error": "no positions to evaluate"}

    ppl = math.exp(-log_prob_sum / n_positions)

    backoff_dist = {}
    for lvl in range(1, ngram_model.max_n + 1):
        count = level_counts.get(lvl, 0)
        backoff_dist[str(lvl)] = round(count / n_positions, 4) if n_positions else 0

    return {
        "perplexity": round(ppl, 2),
        "log_prob_sum": round(log_prob_sum, 2),
        "num_positions": n_positions,
        "oov_count": oov_count,
        "oov_rate": round(oov_count / n_positions, 4) if n_positions else 0,
        "backoff_distribution": backoff_dist,
    }


# ---------------------------------------------------------------------------
#  N-gram top-k
# ---------------------------------------------------------------------------

def compute_ngram_topk(ngram_model, tokens: list, ks=(1, 3, 5)) -> dict:
    hits = {k: 0 for k in ks}
    n_positions = 0

    for i in range(1, len(tokens)):
        context = tokens[:i]
        target = tokens[i]
        cands, _ = get_ngram_distribution(ngram_model, context)

        ranked = sorted(cands, key=cands.get, reverse=True)
        for k in ks:
            if target in ranked[:k]:
                hits[k] += 1
        n_positions += 1

    if n_positions == 0:
        return {f"top{k}": None for k in ks}

    result = {"num_positions": n_positions}
    for k in ks:
        result[f"top{k}"] = round(hits[k] / n_positions, 4)
    return result


# ---------------------------------------------------------------------------
#  Transformer perplexity
# ---------------------------------------------------------------------------

def compute_transformer_perplexity(model, test_ids: list, block_size: int,
                                   device: str) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    ids_tensor = torch.tensor(test_ids, dtype=torch.long)

    with torch.no_grad():
        for start in range(0, len(test_ids) - 1, block_size):
            end = min(start + block_size + 1, len(test_ids))
            chunk = ids_tensor[start:end]
            if len(chunk) < 2:
                break
            x = chunk[:-1].unsqueeze(0).to(device)
            y = chunk[1:].unsqueeze(0).to(device)
            seq_len = x.shape[1]
            if seq_len > block_size:
                x = x[:, :block_size]
                y = y[:, :block_size]
                seq_len = block_size
            _, loss, _ = model(x, targets=y)
            total_loss += loss.item() * seq_len
            total_tokens += seq_len

    if total_tokens == 0:
        return {"perplexity": None, "error": "no tokens to evaluate"}

    mean_loss = total_loss / total_tokens
    ppl = math.exp(mean_loss)

    return {
        "perplexity": round(ppl, 2),
        "mean_cross_entropy": round(mean_loss, 4),
        "num_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
#  Transformer top-k
# ---------------------------------------------------------------------------

def compute_transformer_topk(model, test_ids: list, block_size: int,
                             device: str, ks=(1, 3, 5)) -> dict:
    model.eval()
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    total = 0

    ids_tensor = torch.tensor(test_ids, dtype=torch.long)

    with torch.no_grad():
        for start in range(0, len(test_ids) - 1, block_size):
            end = min(start + block_size + 1, len(test_ids))
            chunk = ids_tensor[start:end]
            if len(chunk) < 2:
                break
            x = chunk[:-1].unsqueeze(0).to(device)
            y = chunk[1:]
            seq_len = x.shape[1]
            if seq_len > block_size:
                x = x[:, :block_size]
                y = y[:block_size]
                seq_len = block_size

            logits, _, _ = model(x)
            logits = logits[0]  # (seq_len, vocab)

            top_indices = torch.topk(logits, max_k, dim=-1).indices  # (seq_len, max_k)
            targets = y.to(device)

            for t in range(seq_len):
                target_id = targets[t].item()
                top_ids = top_indices[t].tolist()
                for k in ks:
                    if target_id in top_ids[:k]:
                        hits[k] += 1
                total += 1

    if total == 0:
        return {f"top{k}": None for k in ks}

    result = {"num_tokens": total}
    for k in ks:
        result[f"top{k}"] = round(hits[k] / total, 4)
    return result


# ---------------------------------------------------------------------------
#  Bits-per-character
# ---------------------------------------------------------------------------

def compute_bits_per_char(perplexity: float, n_tokens: int, n_chars: int) -> float:
    if perplexity is None or n_tokens == 0 or n_chars == 0:
        return None
    return round(math.log2(perplexity) * n_tokens / n_chars, 4)


# ---------------------------------------------------------------------------
#  Latency
# ---------------------------------------------------------------------------

def measure_ngram_latency(ngram_model, prompts: list, n_repeats: int) -> dict:
    timings = []
    token_counts = []

    for prompt in prompts:
        for rep in range(n_repeats + 1):
            t0 = time.perf_counter()
            result = ngram_model.predict_until_sentence_end(
                prompt, max_words=50, temperature=1.0
            )
            elapsed = (time.perf_counter() - t0) * 1000
            if rep > 0:
                timings.append(elapsed)
                token_counts.append(len(result.split()) if result else 0)

    if not timings:
        return {"error": "no timings collected"}

    total_tokens = sum(token_counts)
    total_time_s = sum(timings) / 1000

    return {
        "mean_ms": round(mean(timings), 2),
        "median_ms": round(median(timings), 2),
        "p95_ms": round(sorted(timings)[int(len(timings) * 0.95)], 2),
        "std_ms": round(stdev(timings), 2) if len(timings) > 1 else 0,
        "tokens_per_second": round(total_tokens / total_time_s, 1) if total_time_s > 0 else 0,
        "num_prompts": len(prompts),
        "repeats_per_prompt": n_repeats,
    }


def measure_transformer_latency(engine, prompts: list, n_repeats: int) -> dict:
    timings = []
    token_counts = []

    for prompt in prompts:
        for rep in range(n_repeats + 1):
            t0 = time.perf_counter()
            result = engine.generate_until_sentence_end(
                prompt, max_new_tokens=50
            )
            elapsed = (time.perf_counter() - t0) * 1000
            if rep > 0:
                timings.append(elapsed)
                token_counts.append(len(result.split()) if result else 0)

    if not timings:
        return {"error": "no timings collected"}

    total_tokens = sum(token_counts)
    total_time_s = sum(timings) / 1000

    return {
        "mean_ms": round(mean(timings), 2),
        "median_ms": round(median(timings), 2),
        "p95_ms": round(sorted(timings)[int(len(timings) * 0.95)], 2),
        "std_ms": round(stdev(timings), 2) if len(timings) > 1 else 0,
        "tokens_per_second": round(total_tokens / total_time_s, 1) if total_time_s > 0 else 0,
        "num_prompts": len(prompts),
        "repeats_per_prompt": n_repeats,
    }


# ---------------------------------------------------------------------------
#  Memory
# ---------------------------------------------------------------------------

def measure_ngram_memory(pickle_path: str, prompt: str) -> dict:
    disk_mb = round(os.path.getsize(pickle_path) / (1024 * 1024), 1)

    gc.collect()
    tracemalloc.start()
    from model import NgramPredictor
    m = NgramPredictor(load_from_pickle=pickle_path)
    _, load_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    load_peak_mb = round(load_peak / (1024 * 1024), 1)

    gc.collect()
    tracemalloc.start()
    m.predict_until_sentence_end(prompt, max_words=50)
    _, inf_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    inf_peak_mb = round(inf_peak / (1024 * 1024), 1)

    del m
    gc.collect()

    return {
        "model_load_peak_mb": load_peak_mb,
        "inference_peak_mb": inf_peak_mb,
        "disk_size_mb": disk_mb,
    }


def measure_transformer_memory(ckpt_path: str, tok_path: str,
                               prompt: str) -> dict:
    disk_mb = round(os.path.getsize(ckpt_path) / (1024 * 1024), 1)
    tok_mb = round(os.path.getsize(tok_path) / (1024 * 1024), 2)

    gc.collect()
    tracemalloc.start()
    from transformer.sample import TransformerEngine
    eng = TransformerEngine(ckpt_path, tok_path)
    _, load_peak_py = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    param_bytes = sum(
        p.numel() * p.element_size() for p in eng.model.parameters()
    )
    param_mb = round(param_bytes / (1024 * 1024), 1)
    num_params = eng.model.num_parameters()
    # tracemalloc misses PyTorch C++ allocations; add parameter memory
    load_peak_mb = round(load_peak_py / (1024 * 1024) + param_mb, 1)

    gc.collect()
    tracemalloc.start()
    eng.generate_until_sentence_end(prompt, max_new_tokens=50)
    _, inf_peak_py = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    inf_peak_mb = round(inf_peak_py / (1024 * 1024), 1)

    del eng
    gc.collect()

    return {
        "model_load_peak_mb": load_peak_mb,
        "inference_peak_mb": inf_peak_mb,
        "disk_size_mb": disk_mb,
        "tokenizer_size_mb": tok_mb,
        "param_memory_mb": param_mb,
        "num_parameters": num_params,
    }


# ---------------------------------------------------------------------------
#  Metadata
# ---------------------------------------------------------------------------

def build_metadata(ngram_model, transformer_engine, test_info: dict) -> dict:
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "test_data": test_info,
    }

    if ngram_model is not None:
        meta["ngram_config"] = {
            "max_n": ngram_model.max_n,
            "vocab_size": len(ngram_model.word_counts),
            "total_training_words": ngram_model.total_words,
            "model_file": "saved_model.pkl",
            "model_size_mb": round(
                os.path.getsize(NGRAM_PICKLE) / (1024 * 1024), 1
            ) if os.path.exists(NGRAM_PICKLE) else None,
        }

    if transformer_engine is not None:
        cfg = transformer_engine.cfg
        meta["transformer_config"] = {
            "vocab_size": cfg.vocab_size,
            "block_size": cfg.block_size,
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "d_model": cfg.d_model,
            "d_ff": cfg.d_ff,
            "dropout": cfg.dropout,
            "num_parameters": transformer_engine.model.num_parameters(),
            "model_file": "checkpoints/transformer.pt",
            "model_size_mb": round(
                os.path.getsize(TRANSFORMER_CKPT) / (1024 * 1024), 1
            ) if os.path.exists(TRANSFORMER_CKPT) else None,
            "device": transformer_engine.device,
        }

    return meta


# ---------------------------------------------------------------------------
#  Markdown tables
# ---------------------------------------------------------------------------

def format_markdown_tables(results: dict) -> str:
    lines = []

    # ── Table 1: Model Overview ──
    meta = results.get("metadata", {})
    nc = meta.get("ngram_config", {})
    tc = meta.get("transformer_config", {})

    lines.append("### Table 1 — Model Overview")
    lines.append("")
    lines.append("| Property | N-gram (7-gram) | Transformer (GPT) |")
    lines.append("|---|---|---|")

    n_max = nc.get("max_n", "?")
    t_layers = tc.get("n_layer", "?")
    lines.append(
        f"| Architecture | Stupid Backoff {n_max}->1 "
        f"| {t_layers}-layer decoder-only |"
    )

    nv = nc.get("vocab_size")
    tv = tc.get("vocab_size")
    nv_s = f"{nv:,}" if isinstance(nv, int) else "N/A"
    tv_s = f"{tv:,}" if isinstance(tv, int) else "N/A"
    lines.append(f"| Vocabulary | {nv_s} (whitespace) | {tv_s} (BPE) |")

    tp_count = tc.get("num_parameters", 0)
    tp_s = f"{tp_count / 1e6:.1f}M" if tp_count else "N/A"
    lines.append(f"| Parameters | N/A (count tables) | {tp_s} |")

    ntw = nc.get("total_training_words")
    ntw_s = f"{ntw:,}" if isinstance(ntw, int) else "N/A"
    lines.append(f"| Training tokens | {ntw_s} | trained on Colab |")

    lines.append(
        f"| Disk size | {nc.get('model_size_mb', 'N/A')} MB "
        f"| {tc.get('model_size_mb', 'N/A')} MB |"
    )
    lines.append("")

    # ── Table 2: Quality ──
    np_ = results.get("perplexity", {}).get("ngram", {})
    tp = results.get("perplexity", {}).get("transformer", {})
    na = results.get("topk_accuracy", {}).get("ngram", {})
    ta = results.get("topk_accuracy", {}).get("transformer", {})
    nbpc = results.get("perplexity", {}).get("ngram_bpc")
    tbpc = results.get("perplexity", {}).get("transformer_bpc")

    lines.append("### Table 2 — Language Modeling Quality")
    lines.append("")
    lines.append("| Metric | N-gram | Transformer |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Perplexity | {np_.get('perplexity', 'N/A')} "
        f"| {tp.get('perplexity', 'N/A')} |"
    )
    lines.append(
        f"| Bits-per-character | {nbpc if nbpc else 'N/A'} "
        f"| {tbpc if tbpc else 'N/A'} |"
    )
    def _fmt_pct(v):
        return f"{v:.2%}" if isinstance(v, (int, float)) else "N/A"

    def _fmt_int(v):
        return f"{v:,}" if isinstance(v, int) else "N/A"

    lines.append(f"| Top-1 Accuracy | {_fmt_pct(na.get('top1'))} | {_fmt_pct(ta.get('top1'))} |")
    lines.append(f"| Top-3 Accuracy | {_fmt_pct(na.get('top3'))} | {_fmt_pct(ta.get('top3'))} |")
    lines.append(f"| Top-5 Accuracy | {_fmt_pct(na.get('top5'))} | {_fmt_pct(ta.get('top5'))} |")
    lines.append(f"| OOV Rate | {_fmt_pct(np_.get('oov_rate'))} | N/A (BPE) |")
    lines.append(f"| Eval tokens | {_fmt_int(np_.get('num_positions'))} | {_fmt_int(tp.get('num_tokens'))} |")
    lines.append("")

    # ── Table 3: Latency ──
    nl = results.get("latency", {}).get("ngram", {})
    tl = results.get("latency", {}).get("transformer", {})

    lines.append("### Table 3 — Generation Performance")
    lines.append("")
    lines.append("| Metric | N-gram | Transformer |")
    lines.append("|---|---|---|")
    for key, label in [
        ("mean_ms", "Mean latency (ms)"),
        ("median_ms", "Median latency (ms)"),
        ("p95_ms", "P95 latency (ms)"),
        ("tokens_per_second", "Tokens/second"),
    ]:
        nv = nl.get(key, "N/A")
        tv = tl.get(key, "N/A")
        if isinstance(nv, float):
            nv = f"{nv:,.1f}"
        if isinstance(tv, float):
            tv = f"{tv:,.1f}"
        lines.append(f"| {label} | {nv} | {tv} |")
    lines.append("")

    # ── Table 4: Memory ──
    nm = results.get("memory", {}).get("ngram", {})
    tm = results.get("memory", {}).get("transformer", {})

    lines.append("### Table 4 — Resource Usage")
    lines.append("")
    lines.append("| Metric | N-gram | Transformer |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Model load peak RAM (MB) | {nm.get('model_load_peak_mb', 'N/A')} "
        f"| {tm.get('model_load_peak_mb', 'N/A')} |"
    )
    lines.append(
        f"| Inference peak RAM (MB) | {nm.get('inference_peak_mb', 'N/A')} "
        f"| {tm.get('inference_peak_mb', 'N/A')} |"
    )
    lines.append(
        f"| Disk footprint (MB) | {nm.get('disk_size_mb', 'N/A')} "
        f"| {tm.get('disk_size_mb', 'N/A')} |"
    )
    lines.append(
        f"| Parameter memory (MB) | N/A "
        f"| {tm.get('param_memory_mb', 'N/A')} |"
    )
    lines.append("")

    # ── Table 5: Backoff Distribution ──
    bd = np_.get("backoff_distribution", {})
    if bd:
        lines.append("### Table 5 — N-gram Backoff Level Distribution")
        lines.append("")
        lines.append("| Backoff Level | Frequency |")
        lines.append("|---|---|")
        level_names = {
            "7": "7-gram", "6": "6-gram", "5": "5-gram", "4": "4-gram",
            "3": "Trigram", "2": "Bigram", "1": "Unigram",
        }
        n_pos = np_.get("num_positions", 1)
        for lvl in ["7", "6", "5", "4", "3", "2", "1"]:
            frac = bd.get(lvl, 0)
            count = int(round(frac * n_pos))
            lines.append(
                f"| {level_names.get(lvl, lvl)} | {frac:.1%} ({count:,}) |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-chars", type=int, default=50_000)
    parser.add_argument("--latency-repeats", type=int, default=5)
    parser.add_argument("--latency-prompts", type=int, default=10)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--output", default="evaluation_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    results = {
        "perplexity": {},
        "topk_accuracy": {},
        "latency": {},
        "memory": {},
    }

    # ── 1. Test data ──
    print_progress("Preparing test data...")
    raw_test = prepare_test_data(STORY_PATH, NGRAM_TRAIN_LIMIT, args.test_chars)
    prompts = extract_test_prompts(raw_test, args.latency_prompts)
    test_info = {
        "source": "story.txt",
        "train_cutoff_chars": NGRAM_TRAIN_LIMIT,
        "test_chars_used": len(raw_test),
    }

    # ── 2. Load N-gram ──
    ngram_model = None
    ngram_tokens = []
    if os.path.exists(NGRAM_PICKLE):
        try:
            print_progress("Loading N-gram model...")
            from model import NgramPredictor
            ngram_model = NgramPredictor(load_from_pickle=NGRAM_PICKLE)
            ngram_tokens = tokenize_for_ngram(ngram_model, raw_test)
            test_info["ngram_test_tokens"] = len(ngram_tokens)
            print_progress(f"  N-gram test tokens: {len(ngram_tokens):,}")
        except Exception as e:
            print_progress(f"  WARNING: N-gram load failed: {e}")
    else:
        print_progress("  WARNING: saved_model.pkl not found, skipping N-gram.")

    # ── 3. Load Transformer ──
    transformer_engine = None
    transformer_ids = []
    if os.path.exists(TRANSFORMER_CKPT) and os.path.exists(TOKENIZER_PATH):
        try:
            print_progress("Loading Transformer model...")
            from text_utils import load_tokenizer, encode as bpe_encode
            from transformer.sample import TransformerEngine
            transformer_engine = TransformerEngine(TRANSFORMER_CKPT, TOKENIZER_PATH)
            transformer_ids = bpe_encode(raw_test)
            test_info["transformer_test_tokens"] = len(transformer_ids)
            print_progress(f"  Transformer test tokens: {len(transformer_ids):,}")
        except Exception as e:
            print_progress(f"  WARNING: Transformer load failed: {e}")
    else:
        missing = []
        if not os.path.exists(TRANSFORMER_CKPT):
            missing.append("checkpoints/transformer.pt")
        if not os.path.exists(TOKENIZER_PATH):
            missing.append("tokenizer/spm.model")
        print_progress(f"  WARNING: Missing {', '.join(missing)}, skipping Transformer.")

    results["metadata"] = build_metadata(ngram_model, transformer_engine, test_info)

    # ── 4. N-gram perplexity ──
    if ngram_model and ngram_tokens:
        try:
            print_progress("Computing N-gram perplexity...")
            results["perplexity"]["ngram"] = compute_ngram_perplexity(
                ngram_model, ngram_tokens
            )
            print_progress(
                f"  Perplexity: {results['perplexity']['ngram']['perplexity']}"
            )
        except Exception as e:
            print_progress(f"  ERROR: N-gram perplexity failed: {e}")
            results["perplexity"]["ngram"] = {"error": str(e)}

    # ── 5. Transformer perplexity ──
    if transformer_engine and transformer_ids:
        try:
            print_progress("Computing Transformer perplexity...")
            results["perplexity"]["transformer"] = compute_transformer_perplexity(
                transformer_engine.model,
                transformer_ids,
                transformer_engine.cfg.block_size,
                transformer_engine.device,
            )
            print_progress(
                f"  Perplexity: {results['perplexity']['transformer']['perplexity']}"
            )
        except Exception as e:
            print_progress(f"  ERROR: Transformer perplexity failed: {e}")
            results["perplexity"]["transformer"] = {"error": str(e)}

    # ── 6. Bits-per-character ──
    n_chars = len(raw_test)
    np_data = results["perplexity"].get("ngram", {})
    tp_data = results["perplexity"].get("transformer", {})

    ngram_bpc = compute_bits_per_char(
        np_data.get("perplexity"),
        np_data.get("num_positions", 0),
        n_chars,
    )
    transformer_bpc = compute_bits_per_char(
        tp_data.get("perplexity"),
        tp_data.get("num_tokens", 0),
        n_chars,
    )
    results["perplexity"]["ngram_bpc"] = ngram_bpc
    results["perplexity"]["transformer_bpc"] = transformer_bpc
    if ngram_bpc:
        print_progress(f"  N-gram BPC: {ngram_bpc}")
    if transformer_bpc:
        print_progress(f"  Transformer BPC: {transformer_bpc}")

    # ── 7. N-gram top-k ──
    if ngram_model and ngram_tokens:
        try:
            print_progress("Computing N-gram top-k accuracy...")
            results["topk_accuracy"]["ngram"] = compute_ngram_topk(
                ngram_model, ngram_tokens
            )
            a = results["topk_accuracy"]["ngram"]
            print_progress(
                f"  Top-1: {a['top1']:.2%}  Top-3: {a['top3']:.2%}  "
                f"Top-5: {a['top5']:.2%}"
            )
        except Exception as e:
            print_progress(f"  ERROR: N-gram top-k failed: {e}")
            results["topk_accuracy"]["ngram"] = {"error": str(e)}

    # ── 8. Transformer top-k ──
    if transformer_engine and transformer_ids:
        try:
            print_progress("Computing Transformer top-k accuracy...")
            results["topk_accuracy"]["transformer"] = compute_transformer_topk(
                transformer_engine.model,
                transformer_ids,
                transformer_engine.cfg.block_size,
                transformer_engine.device,
            )
            a = results["topk_accuracy"]["transformer"]
            print_progress(
                f"  Top-1: {a['top1']:.2%}  Top-3: {a['top3']:.2%}  "
                f"Top-5: {a['top5']:.2%}"
            )
        except Exception as e:
            print_progress(f"  ERROR: Transformer top-k failed: {e}")
            results["topk_accuracy"]["transformer"] = {"error": str(e)}

    # ── 9. Latency ──
    if not args.skip_latency:
        if ngram_model:
            try:
                print_progress(
                    f"Measuring N-gram latency ({len(prompts)} prompts "
                    f"x {args.latency_repeats} repeats)..."
                )
                results["latency"]["ngram"] = measure_ngram_latency(
                    ngram_model, prompts, args.latency_repeats
                )
                print_progress(
                    f"  Mean: {results['latency']['ngram']['mean_ms']:.1f} ms"
                )
            except Exception as e:
                print_progress(f"  ERROR: N-gram latency failed: {e}")
                results["latency"]["ngram"] = {"error": str(e)}

        if transformer_engine:
            try:
                print_progress(
                    f"Measuring Transformer latency ({len(prompts)} prompts "
                    f"x {args.latency_repeats} repeats)..."
                )
                results["latency"]["transformer"] = measure_transformer_latency(
                    transformer_engine, prompts, args.latency_repeats
                )
                print_progress(
                    f"  Mean: {results['latency']['transformer']['mean_ms']:.1f} ms"
                )
            except Exception as e:
                print_progress(f"  ERROR: Transformer latency failed: {e}")
                results["latency"]["transformer"] = {"error": str(e)}
    else:
        print_progress("Skipping latency measurement (--skip-latency)")

    # ── 10. Memory ──
    if not args.skip_memory:
        print_progress("Measuring memory usage (fresh model loads)...")
        test_prompt = prompts[0] if prompts else "the history of"

        del ngram_model
        del transformer_engine
        gc.collect()

        if os.path.exists(NGRAM_PICKLE):
            try:
                print_progress("  Measuring N-gram memory...")
                results["memory"]["ngram"] = measure_ngram_memory(
                    NGRAM_PICKLE, test_prompt
                )
                print_progress(
                    f"  N-gram load peak: "
                    f"{results['memory']['ngram']['model_load_peak_mb']} MB"
                )
            except Exception as e:
                print_progress(f"  ERROR: N-gram memory failed: {e}")
                results["memory"]["ngram"] = {"error": str(e)}

        if os.path.exists(TRANSFORMER_CKPT) and os.path.exists(TOKENIZER_PATH):
            try:
                print_progress("  Measuring Transformer memory...")
                results["memory"]["transformer"] = measure_transformer_memory(
                    TRANSFORMER_CKPT, TOKENIZER_PATH, test_prompt
                )
                print_progress(
                    f"  Transformer load peak: "
                    f"{results['memory']['transformer']['model_load_peak_mb']} MB"
                )
            except Exception as e:
                print_progress(f"  ERROR: Transformer memory failed: {e}")
                results["memory"]["transformer"] = {"error": str(e)}
    else:
        print_progress("Skipping memory measurement (--skip-memory)")

    # ── 11. Output ──
    tables = format_markdown_tables(results)
    results["markdown_tables"] = tables

    print()
    print("=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)
    print()
    print(tables)

    output_path = args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print_progress(f"Results saved to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Partial results may not be saved.")
        sys.exit(1)
