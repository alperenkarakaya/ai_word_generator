"""
Flask API — N-gram + Transformer text generation.

Endpoints
---------
GET /                    Web UI
GET /predict_sentence    Sentence completion  (Shift+Tab)
GET /predict_paragraph   Paragraph generation (Ctrl+Shift+Tab)
GET /probabilities       N-gram probability tables for the UI panels
GET /health              JSON status

Query param ?engine=ngram (default) | transformer
"""

import os
from flask import Flask, render_template, request, jsonify

# ── paths from environment (override for deployment) ─────────────────────────
TRANSFORMER_CKPT = os.environ.get("TRANSFORMER_CKPT", "checkpoints/transformer.pt")
TOKENIZER_PATH   = os.environ.get("TOKENIZER_PATH",   "tokenizer/spm.model")

app = Flask(__name__)

# ── N-gram model (primary generation engine) ─────────────────────────────────
ngram = None
ngram_error = None

if os.path.exists("saved_model.pkl"):
    try:
        from model import NgramPredictor
        ngram = NgramPredictor(load_from_pickle="saved_model.pkl")
    except Exception as exc:
        ngram_error = str(exc)
        print(f"WARNING: N-gram failed to load — {exc}")
else:
    ngram_error = "saved_model.pkl not found. Run create_pickle.py first."
    print(f"INFO: {ngram_error}")

# ── Transformer model (optional alternative engine) ───────────────────────────
transformer = None
transformer_error = None

if os.path.exists(TRANSFORMER_CKPT) and os.path.exists(TOKENIZER_PATH):
    try:
        from transformer.sample import TransformerEngine
        transformer = TransformerEngine(TRANSFORMER_CKPT, TOKENIZER_PATH)
    except Exception as exc:
        transformer_error = str(exc)
        print(f"WARNING: Transformer failed to load — {exc}")
else:
    transformer_error = (
        f"Checkpoint not found ({TRANSFORMER_CKPT}). "
        "Train on Colab first — see notebooks/colab_train.ipynb"
    )
    print(f"INFO: {transformer_error}")


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template(
        "index.html",
        ngram_ready=(ngram is not None),
        ngram_error=ngram_error,
        ngram_max_n=getattr(ngram, "max_n", 0) if ngram else 0,
        transformer_ready=(transformer is not None),
        transformer_error=transformer_error,
    )


@app.route("/predict_sentence")
def predict_sentence():
    """
    Sentence completion (Shift+Tab).

    Query params
    ------------
    text       : str  — current editor content
    engine     : str  — "ngram" (default) or "transformer"
    max_tokens : int  — max new tokens / words (default 100)
    """
    engine     = request.args.get("engine", "ngram")
    text       = request.args.get("text", "")
    max_tokens = int(request.args.get("max_tokens", 100))

    if engine == "transformer":
        if transformer is None:
            return jsonify({
                "error": transformer_error or "Transformer not loaded.",
                "hint":  "Train the model on Colab and place checkpoints/transformer.pt here.",
            }), 503
        try:
            completion = transformer.generate_until_sentence_end(text, max_new_tokens=max_tokens)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
        return jsonify({"completion": completion})

    # default: ngram
    if ngram is None:
        return jsonify({
            "error": ngram_error or "N-gram model not loaded.",
            "hint":  "Run create_pickle.py to build the model first.",
        }), 503
    try:
        completion = ngram.predict_until_sentence_end(text, max_words=max_tokens)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify({"completion": completion})


@app.route("/predict_paragraph")
def predict_paragraph():
    """
    Paragraph generation (Ctrl+Shift+Tab).

    Query params
    ------------
    text          : str — prompt
    engine        : str — "ngram" (default) or "transformer"
    max_sentences : int — number of sentences (default 5)
    """
    engine        = request.args.get("engine", "ngram")
    text          = request.args.get("text", "")
    max_sentences = int(request.args.get("max_sentences", 5))

    if engine == "transformer":
        if transformer is None:
            return jsonify({"error": transformer_error}), 503
        try:
            paragraph = transformer.generate_paragraph(text, max_sentences=max_sentences)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
        return jsonify({"completion": paragraph})

    # default: ngram
    if ngram is None:
        return jsonify({
            "error": ngram_error or "N-gram model not loaded.",
            "hint":  "Run create_pickle.py first.",
        }), 503
    try:
        paragraph = ngram.predict_paragraph(text, max_sentences=max_sentences)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify({"completion": paragraph})


@app.route("/predict_next")
def predict_next():
    """
    Returns the full ghost-layer text for the inline word suggestion.
    Called on every keystroke; always uses the N-gram engine.

    Query params
    ------------
    text : str — current editor content
    """
    if ngram is None:
        return jsonify({"ghost": ""})
    text  = request.args.get("text", "")
    ghost = ngram.get_ghost_text(text)
    return jsonify({"ghost": ghost})


@app.route("/probabilities")
def get_probabilities():
    """Unigram / bigram / trigram probability tables for the UI panels."""
    if ngram is None:
        return jsonify({"unigram": [], "bigram": [], "trigram": [],
                        "current_word": "", "context": ""})
    text       = request.args.get("text", "")
    use_tokens = request.args.get("use_tokens", "true").lower() == "true"
    return jsonify(ngram.get_probabilities(text, use_tokens=use_tokens))


@app.route("/health")
def health():
    return jsonify({
        "ngram": {
            "loaded": ngram is not None,
            "max_n":  getattr(ngram, "max_n", None),
            "vocab":  len(ngram.word_counts) if ngram else 0,
            "error":  ngram_error,
        },
        "transformer": {
            "loaded":     transformer is not None,
            "checkpoint": TRANSFORMER_CKPT,
            "error":      transformer_error,
        },
    })


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  AI Text Generator")
    print("=" * 60)
    if ngram:
        print(f"  N-gram     : OK  (max_n={ngram.max_n}, vocab={len(ngram.word_counts):,})")
    else:
        print(f"  N-gram     : NOT READY  ({ngram_error})")
    if transformer:
        print("  Transformer: OK")
    else:
        print(f"  Transformer: not loaded  ({transformer_error})")
    print()
    print("  http://localhost:5000")
    print("=" * 60)
    print()
    app.run(debug=True, host="0.0.0.0", port=5000)
