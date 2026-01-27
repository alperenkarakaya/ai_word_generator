"""
Flask API (Noktalama Token Desteği ile)
"""
from flask import Flask, render_template, request, jsonify
from model import NgramPredictor

app = Flask(__name__)

# Model yükle
print("📦 Model yükleniyor...")
try:
    # LSTM model varsa hybrid mode
    lstm_path = "lstm_model.pth"
    ai = NgramPredictor(load_from_pickle="saved_model.pkl", lstm_model_path=lstm_path)
    print("✅ Model hazır!\n")
except Exception as e:
    print(f"❌ Hata: {e}")
    ai = None


@app.route('/')
def home():
    if ai is None:
        return "<h1>❌ Model yüklenemedi</h1>", 500
    return render_template('index.html')


@app.route('/predict')
def get_prediction():
    if ai is None:
        return jsonify({'error': 'Model yok'}), 500
    
    text = request.args.get('text', '')
    use_tokens = request.args.get('use_tokens', 'true').lower() == 'true'
    
    prediction = ai.predict(text, use_tokens=use_tokens)
    
    return jsonify({
        'prediction': prediction,
        'use_tokens': use_tokens
    })


@app.route('/probabilities')
def get_probabilities():
    if ai is None:
        return jsonify({'error': 'Model yok'}), 500
    
    text = request.args.get('text', '')
    use_tokens = request.args.get('use_tokens', 'true').lower() == 'true'
    
    probs = ai.get_probabilities(text, use_tokens=use_tokens)
    
    return jsonify(probs)


@app.route('/predict_sentence')
def predict_sentence():
    """Cümle tamamlama endpoint'i (Shift+Tab için)"""
    if ai is None:
        return jsonify({'error': 'Model yok'}), 500
    
    text = request.args.get('text', '')
    use_tokens = request.args.get('use_tokens', 'true').lower() == 'true'
    max_words = int(request.args.get('max_words', '50'))
    use_lstm = request.args.get('use_lstm', 'false').lower() == 'true'
    
    # LSTM model varsa ve isteniyorsa onu kullan
    if use_lstm and ai.lstm_model:
        try:
            completion = ai.lstm_model.predict(text, use_tokens=use_tokens)
        except Exception as e:
            print(f"LSTM hatası: {e}, N-gram'a geçiliyor...")
            completion = ai.predict_until_sentence_end(text, use_tokens=use_tokens, max_words=max_words)
    else:
        # N-gram ile cümle bitirici tokena kadar devam et
        completion = ai.predict_until_sentence_end(text, use_tokens=use_tokens, max_words=max_words)
    
    return jsonify({
        'completion': completion,
        'use_tokens': use_tokens,
        'model_used': 'lstm' if (use_lstm and ai.lstm_model) else 'ngram'
    })


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok' if ai else 'error',
        'model_loaded': ai is not None,
        'features': {
            'punctuation_tokens': True,
            'cross_sentence': True
        }
    })


if __name__ == '__main__':
    if ai:
        print("🚀 Flask başlatılıyor: http://localhost:5000")
        print("📍 Noktalama token sistemi: AKTİF\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Model yüklenemedi!")