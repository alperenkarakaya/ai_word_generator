from flask import Flask, render_template, request, jsonify
from model import NgramPredictor

app = Flask(__name__, 
            static_folder='static',
            static_url_path='/static',
            template_folder='templates')

# ✅ Pickle'dan hızlıca yükle
print("📦 Model yükleniyor...")
try:
    ai = NgramPredictor(load_from_pickle="saved_model.pkl")
    print("✅ Model hazır!\n")
except FileNotFoundError:
    print("❌ saved_model.pkl bulunamadı!")
    print("   Önce modeli eğitin: python model.py train\n")
    ai = None


@app.route('/')
def home():
    """Ana sayfa"""
    if ai is None:
        return "<h1>❌ Model Yüklenemedi</h1><p>Lütfen 'python model.py train' komutunu çalıştırın.</p>", 500
    return render_template('index.html')


@app.route('/predict')
def get_prediction():
    """Kelime/cümle tamamlama endpoint'i"""
    if ai is None:
        return jsonify({'error': 'Model yüklenemedi'}), 500
    
    text = request.args.get('text', '')
    mode = request.args.get('mode', 'suffix')
    use_transformer = request.args.get('use_transformer', 'false').lower() == 'true'
    max_words = int(request.args.get('max_words', 6))
    
    try:
        if mode == 'completion':
            if use_transformer:
                prediction = ai.predict(text, mode='completion', use_transformer=True)
            else:
                prediction = ai._complete_sentence(text, max_words=max_words)
        else:
            prediction = ai.predict(text, mode='suffix')
        
        return jsonify({
            'prediction': prediction,
            'mode': mode,
            'model_used': 'transformer' if use_transformer and ai.transformer_model else 'ngram'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/probabilities')
def get_probabilities():
    """Olasılık hesaplama endpoint'i"""
    if ai is None:
        return jsonify({'error': 'Model yüklenemedi'}), 500
    
    text = request.args.get('text', '')
    probabilities = ai.get_probabilities(text)
    return jsonify(probabilities)


@app.route('/health')
def health_check():
    """Sağlık kontrolü ve model durumu"""
    return jsonify({
        'status': 'ok' if ai else 'error',
        'ngram_loaded': ai is not None,
        'transformer_loaded': ai.transformer_model is not None if ai else False,
        'device': ai.device if ai else 'N/A'
    })


if __name__ == '__main__': 
    import os
    port = int(os.environ.get('PORT', 5000))
    
    if ai:
        print(f"🚀 Flask başlatılıyor: http://localhost:{port}\n")
        app.run(debug=True, host='0.0.0.0', port=port)
    else:
        print("❌ Model yüklenemedi, Flask başlatılamadı.")
        print("   Önce: python model.py train")