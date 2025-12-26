from flask import Flask, request, jsonify, render_template_string
from collections import Counter, defaultdict
import os
import re
import random  # Şans faktörü için gerekli

app = Flask(__name__)

# --- 1. MODEL (Weighted/Probabilistic Logic) ---
class CreativePredictor:
    def __init__(self, filename):
        self.word_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.load_data(filename)

    def load_data(self, filename):
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as file:
                text = file.read().lower()
            
            text_clean = re.sub(r'[^\w\s]', '', text)
            words = text_clean.split()
            
            self.word_counts = Counter(words)
            
            for i in range(len(words) - 1):
                current_w = words[i]
                next_w = words[i+1]
                self.bigram_counts[current_w][next_w] += 1
                
            print(f"--- Yaratıcı Model Hazır ---")
        else:
            print("HATA: story.txt bulunamadı.")

    def predict(self, full_text):
        if not full_text or full_text.endswith(" "):
            return ""

        # Son yazılan yarım kelimeyi al
        current_partial = full_text.split()[-1].lower()
        
        # --- ADIM 1: MEVCUT KELİMEYİ TAMAMLA (Hala En İyisini Seçiyor) ---
        # Burası "max" kalmalı, çünkü "digi" yazdıysan "digital" demek istemişsindir.
        # Burada rastgelelik olursa kullanıcı sinir olur.
        matches = [w for w in self.word_counts if w.startswith(current_partial)]
        
        if not matches:
            return ""

        # Kelimeyi tamamla (Deterministik)
        best_completion = max(matches, key=lambda w: self.word_counts[w])
        suffix = best_completion[len(current_partial):]
        
        # --- ADIM 2: SONRAKİ KELİMEYİ TAHMİN ET (Burada Çeşitlilik Var!) ---
        next_word_candidates = self.bigram_counts[best_completion]
        
        prediction_string = suffix
        
        if next_word_candidates:
            # Aday kelimeleri ve sayılarını (ağırlıklarını) ayır
            words = list(next_word_candidates.keys())
            counts = list(next_word_candidates.values())
            
            # BURASI SİHİRİN OLDUĞU YER:
            # random.choices, sayıları "ağırlık" olarak kullanır.
            # Sayısı çok olanın gelme şansı yüksektir ama garanti değildir.
            chosen_next_word = random.choices(words, weights=counts, k=1)[0]
            
            prediction_string += " " + chosen_next_word
            
        return prediction_string

# Modeli Başlat
ai = CreativePredictor("story.txt")

# --- 2. ARAYÜZ (AYNI) ---
HTML_CODE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>AI Creative Writer</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background: #202124; color: white; margin: 0;}
        .container { position: relative; width: 700px; }

        .layer {
            font-family: 'Courier New', Courier, monospace;
            font-size: 24px;
            width: 100%;
            padding: 20px;
            box-sizing: border-box;
            border-radius: 8px;
            border: 1px solid #5f6368;
            line-height: 1.6;
            min-height: 200px;
            resize: none;
            overflow: hidden;
            background: transparent;
        }

        #ghost-input {
            position: absolute;
            top: 0; left: 0;
            z-index: 1;
            color: #9aa0a6; 
            pointer-events: none;
            white-space: pre-wrap;
            border-color: transparent;
        }

        #user-input {
            position: relative;
            z-index: 2;
            color: #e8eaed;
            background: rgba(32, 33, 36, 0.6);
            outline: none;
        }
        #user-input:focus { border-color: #8ab4f8; }

        .hint { margin-top: 15px; text-align: center; color: #9aa0a6; font-size: 0.9em; }
        kbd { background: #3c4043; padding: 4px 8px; border-radius: 4px; border: 1px solid #5f6368; }
    </style>
</head>
<body>
    <div class="container">
        <div id="ghost-input" class="layer"></div>
        <textarea id="user-input" class="layer" placeholder="Yazmaya başla..." autofocus></textarea>
        <div class="hint">Öneri sürekli değişebilir (Variety Mode). Kabul etmek için <kbd>TAB</kbd></div>
    </div>

    <script>
        const inputField = document.getElementById('user-input');
        const ghostField = document.getElementById('ghost-input');
        let currentSuggestion = "";

        inputField.addEventListener('input', async function() {
            const text = this.value;
            
            if (text.length === 0 || text.endsWith(" ")) {
                ghostField.textContent = "";
                currentSuggestion = "";
                return;
            }

            const response = await fetch('/predict?text=' + encodeURIComponent(text));
            const data = await response.json();
            
            currentSuggestion = data.prediction;

            if (currentSuggestion) {
                ghostField.textContent = text + currentSuggestion;
            } else {
                ghostField.textContent = "";
            }
        });

        inputField.addEventListener('keydown', function(e) {
            if (e.key === 'Tab') {
                e.preventDefault();
                if (currentSuggestion) {
                    this.value += currentSuggestion + " ";
                    this.dispatchEvent(new Event('input'));
                }
            }
        });
        
        inputField.addEventListener('scroll', function() {
            ghostField.scrollTop = this.scrollTop;
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_CODE)

@app.route('/predict')
def get_prediction():
    text = request.args.get('text', '')
    prediction = ai.predict(text)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)