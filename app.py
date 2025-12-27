from flask import Flask, render_template, request, jsonify
from model import NgramPredictor

# Static ve template klasörlerini açıkça belirt
app = Flask(__name__, 
            static_folder='static',
            static_url_path='/static',
            template_folder='templates')

# Modeli başlat
ai = NgramPredictor("story.txt")


@app.route('/')
def home():
    """Ana sayfa"""
    return render_template('index.html')


@app.route('/predict')
def get_prediction():
    """Kelime tamamlama endpoint'i"""
    text = request.args.get('text', '')
    prediction = ai.predict(text)
    return jsonify({'prediction': prediction})


@app.route('/probabilities')
def get_probabilities():
    """Olasılık hesaplama endpoint'i"""
    text = request.args.get('text', '')
    probabilities = ai.get_probabilities(text)
    return jsonify(probabilities)


if __name__ == '__main__': 
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)