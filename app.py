"""
Flask API — Transformer-powered text generation.

Generation engine : transformer/sample.py  (TransformerEngine)
Probability panels: model.py               (NgramPredictor, statistics only)

Endpoints
---------
GET /                         Web UI
GET /predict_sentence         Sentence completion (Shift+Tab)
GET /predict_paragraph        Paragraph generation  (Phase 3)
GET /probabilities            N-gram probability table for the UI panels
GET /health                   JSON status
"""

import os

from flask import Flask, render_template, request, jsonify

TRANSFORMER_CKPT = os.environ.get("TRANSFORMER_CKPT", "checkpoints/transformer.pt")
TOKENIZER_PATH   = os.environ.get("TOKENIZER_PATH",   "tokenizer/spm.model")

app = Flask(__name__)

# ── Transformer (primary generation engine) ──────────────────────────────────
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
        f"Train the model on Colab first — see notebooks/colab_train.ipynb"
    )
    print(f"INFO: {transformer_error}")

# ── N-gram (probability display panels only) ──────────────────────────────────
ngram = None
if os.path.exists("saved_model.pkl"):
    try:
        from model import NgramPredictor
        ngram = NgramPredictor(load_from_pickle="saved_model.pkl")
        print("N-gram model loaded (probability panels only).")
    except Exception as exc:
        print(f"WARNING: N-gram failed to load — {exc}")


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template(
        "index.html",
        transformer_ready=(transformer is not None),
        transformer_error=transformer_error,
    )


@app.route("/predict_sentence")
def predict_sentence():
    """
    Sentence completion triggered by Shift+Tab.

    Query params
    ------------
    text       : str  — the current editor content
    max_tokens : int  — max new tokens (default 100)
    """
    if transformer is None:
        return jsonify({
            "error": transformer_error or "Transformer not loaded.",
            "hint": "Train the model on Colab and place checkpoints/transformer.pt here.",
        }), 503

    text       = request.args.get("text", "")
    max_tokens = int(request.args.get("max_tokens", 100))

    try:
        completion = transformer.generate_until_sentence_end(
            text, max_new_tokens=max_tokens
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({"completion": completion})


@app.route("/predict_paragraph")
def predict_paragraph():
    """
    Paragraph generation (Phase 3).

    Query params
    ------------
    text          : str — prompt
    max_sentences : int — stop after this many sentence-enders (default 5)
    """
    if transformer is None:
        return jsonify({"error": transformer_error}), 503

    text          = request.args.get("text", "")
    max_sentences = int(request.args.get("max_sentences", 5))

    try:
        paragraph = transformer.generate_paragraph(text, max_sentences=max_sentences)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({"completion": paragraph})


@app.route("/probabilities")
def get_probabilities():
    """
    Returns unigram / bigram / trigram probability tables for the UI panels.
    Backed by the N-gram model (fast, no GPU needed).
    """
    if ngram is None:
        return jsonify({"unigram": [], "bigram": [], "trigram": [], "current_word": ""})

    text       = request.args.get("text", "")
    use_tokens = request.args.get("use_tokens", "true").lower() == "true"
    return jsonify(ngram.get_probabilities(text, use_tokens=use_tokens))


@app.route("/health")
def health():
    return jsonify({
        "transformer": {
            "loaded": transformer is not None,
            "error":  transformer_error,
            "checkpoint": TRANSFORMER_CKPT,
        },
        "ngram": {
            "loaded": ngram is not None,
            "role": "probability panels only",
        },
    })


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  AI Text Generator — Transformer Edition")
    print("=" * 60)
    if transformer:
        print("  Generation : Transformer  OK")
    else:
        print(f"  Generation : NOT READY  ({transformer_error})")
    print(f"  Prob panels: {'N-gram  OK' if ngram else 'N-gram  -- (run create_pickle.py)'}")
    print()
    print("  http://localhost:5000")
    print("=" * 60)
    print()
    app.run(debug=True, host="0.0.0.0", port=5000)
