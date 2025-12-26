from flask import Flask, request, jsonify, render_template_string
from collections import Counter
import os
import re

app = Flask(__name__)

# --- 1. YAPAY ZEKA MODELİ (Strict Autocomplete) ---
class StrictPredictor:
    def __init__(self, filename):
        self.word_counts = Counter()
        self.load_data(filename)

    def load_data(self, filename):
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as file:
                text = file.read().lower()
            
            # 1. Metni temizle (noktalama işaretlerini boşluğa çevir)
            # Böylece "future." kelimesini "future" olarak öğrenir.
            text_clean = re.sub(r'[^\w\s]', '', text)
            
            # 2. Kelimeleri ayır ve say
            words = text_clean.split()
            self.word_counts = Counter(words)
            
            print("--- Model Hazır ---")
            print(f"Öğrenilen benzersiz kelime sayısı: {len(self.word_counts)}")
        else:
            print("HATA: story.txt bulunamadı.")

    def predict(self, full_text):
        """
        Kullanıcının yazdığı metnin SON kelimesini alır
        ve hikayede o harflerle başlayan en popüler kelimeyi bulur.
        """
        if not full_text:
            return ""

        # Sadece son kelimeye odaklan (Örn: "the future of is" -> "is")
        # Eğer son karakter boşluksa henüz yeni kelimeye başlanmamıştır.
        if full_text.endswith(" "):
            return ""
            
        current_word_part = full_text.split()[-1].lower()
        
        # Henüz bir şey yazılmadıysa dönme
        if not current_word_part:
            return ""

        # --- ARAMA ALGORİTMASI ---
        # Hikayede geçen kelimelerden, yazdığımız kısımla başlayanları bul.
        matches = [word for word in self.word_counts if word.startswith(current_word_part)]
        
        if not matches:
            return ""

        # Eşleşenler arasından hikayede EN ÇOK GEÇENİ seç (Frekans)
        best_match = max(matches, key=lambda w: self.word_counts[w])
        
        # Sadece eksik kalan kısmı döndür
        # Örn: Yazılan="fut", Bulunan="future" -> Döndür="ure"
        return best_match[len(current_word_part):]

# Modeli Başlat
ai = StrictPredictor("story.txt")

# --- 2. HTML ARAYÜZÜ (Gölge Efekti Aynı Kalıyor) ---
HTML_CODE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>Akıllı Tamamlayıcı</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background: #282c34; color: white; margin: 0;}
        .container { position: relative; width: 700px; }

        /* Ortak Stil */
        .layer {
            font-family: 'Courier New', Courier, monospace;
            font-size: 22px;
            width: 100%;
            padding: 20px;
            box-sizing: border-box;
            border-radius: 12px;
            border: 2px solid #444;
            line-height: 1.6;
            min-height: 200px;
            resize: none;
            overflow: hidden;
            background: transparent;
        }

        /* Arka Katman (Gölge/Öneri) */
        #ghost-input {
            position: absolute;
            top: 0; left: 0;
            z-index: 1;
            color: #6c757d; /* Silik Gri */
            pointer-events: none;
            white-space: pre-wrap;
            border-color: transparent; /* Sınırları gizle ki çakışmasın */
        }

        /* Ön Katman (Kullanıcı) */
        #user-input {
            position: relative;
            z-index: 2;
            color: #e6e6e6;
            background: rgba(40, 44, 52, 0.6); /* Hafif koyuluk */
            outline: none;
            transition: border-color 0.3s;
        }
        #user-input:focus { border-color: #61dafb; }

        .hint { margin-top: 15px; text-align: center; color: #aaa; font-size: 0.9em; }
        kbd { background: #444; padding: 3px 8px; border-radius: 4px; color: #61dafb; border: 1px solid #555; }
    </style>
</head>
<body>
    <div class="container">
        <div id="ghost-input" class="layer"></div>
        <textarea id="user-input" class="layer" placeholder="Hikayeden bir kelime yazmaya başla..." autofocus></textarea>
        <div class="hint">Öneriyi kabul etmek için <kbd>TAB</kbd> tuşuna bas.</div>
    </div>

    <script>
        const inputField = document.getElementById('user-input');
        const ghostField = document.getElementById('ghost-input');
        let currentSuggestion = "";

        inputField.addEventListener('input', async function() {
            const text = this.value;
            
            // Metin boşsa veya son karakter boşluksa gölgeyi sil
            if (text.length === 0 || text.endsWith(" ")) {
                ghostField.textContent = "";
                currentSuggestion = "";
                return;
            }

            // Backend'e sor
            const response = await fetch('/predict?text=' + encodeURIComponent(text));
            const data = await response.json();
            
            currentSuggestion = data.prediction;

            // Gölgeyi oluştur: Yazılan Metin + Öneri
            if (currentSuggestion) {
                ghostField.textContent = text + currentSuggestion;
            } else {
                ghostField.textContent = ""; // Öneri yoksa gölgeyi gösterme
            }
        });

        // TAB Tuşu Yakalama
        inputField.addEventListener('keydown', function(e) {
            if (e.key === 'Tab') {
                e.preventDefault();
                if (currentSuggestion) {
                    this.value += currentSuggestion;
                    // İmleci sona taşı ve olayı tetikle
                    this.dispatchEvent(new Event('input'));
                }
            }
        });
        
        // Scroll Senkronizasyonu
        inputField.addEventListener('scroll', function() {
            ghostField.scrollTop = this.scrollTop;
        });
    </script>
</body>
</html>
"""

# --- 3. ROTALAR ---
@app.route('/')
def home():
    return render_template_string(HTML_CODE)

@app.route('/predict')
def get_prediction():
    text = request.args.get('text', '')
    prediction = ai.predict(text)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, port=5001)