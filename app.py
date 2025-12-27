from flask import Flask, request, jsonify, render_template_string
from collections import Counter, defaultdict
import os
import re
import random

app = Flask(__name__)

# --- TRIGRAM MODEL (2 kelimeye bakarak 3. kelimeyi tahmin eder) ---
class TrigramPredictor:  
    def __init__(self, filename):
        self.word_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.trigram_counts = defaultdict(Counter)
        self.load_data(filename)

    def load_data(self, filename):
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as file:
                text = file.read().lower()
            
            text_clean = re.sub(r'[^\w\s]', '', text)
            words = text_clean.split()
            
            self.word_counts = Counter(words)
            
            # Bigram (2-gram:  1 kelime -> sonraki kelime)
            for i in range(len(words) - 1):
                current_w = words[i]
                next_w = words[i+1]
                self.bigram_counts[current_w][next_w] += 1
            
            # TRIGRAM (3-gram: 2 kelime -> sonraki kelime)
            for i in range(len(words) - 2):
                word1 = words[i]
                word2 = words[i+1]
                next_word = words[i+2]
                self.trigram_counts[(word1, word2)][next_word] += 1
                
            print(f"--- Model Hazır ---")
            print(f"Öğrenilen Bigram Sayısı: {len(self.bigram_counts)}")
            print(f"Öğrenilen Trigram Sayısı:  {len(self.trigram_counts)}")
        else:
            print("HATA: story.txt bulunamadı.")

    def get_probabilities(self, full_text):
        """
        Kullanıcının yazdığı metne göre hem 2-gram hem 3-gram olasılıklarını döndürür. 
        SADECE yazdığı kelime için, tamamlama yapmadan. 
        """
        if not full_text: 
            return {"bigram": [], "trigram": [], "current_word": ""}

        words_list = full_text.split()
        
        # Şu anki kelimeyi belirle (tamamlama YAPMA!)
        if full_text.endswith(" "):
            # Boşlukla bittiyse son tamamlanmış kelimeyi al
            if len(words_list) < 1:
                return {"bigram":  [], "trigram": [], "current_word": ""}
            current_word = words_list[-1].lower()
        else:
            # Hala yazıyorsa, yazdığı kelimeyi aynen al
            current_word = words_list[-1].lower()
        
        result = {
            "bigram": [],
            "trigram": [],
            "current_word": current_word
        }
        
        # --- 2-GRAM (BIGRAM) OLASИЛIKLARI ---
        # Sadece son kelimeye bakarak sonraki kelimeyi tahmin et
        bigram_candidates = self.bigram_counts. get(current_word)
        if bigram_candidates:
            result["bigram"] = self._format_probabilities(bigram_candidates)
        
        # --- 3-GRAM (TRIGRAM) OLASИЛIKLARI ---
        # Son 2 kelimeye bakarak sonraki kelimeyi tahmin et
        if len(words_list) >= 2:
            if full_text.endswith(" "):
                # "the future " -> prev="the", current="future"
                prev_word = words_list[-2].lower()
            else:
                # "the future" -> prev="the", current="future"
                if len(words_list) >= 2:
                    prev_word = words_list[-2].lower()
                else:
                    prev_word = None
            
            if prev_word:
                trigram_candidates = self.trigram_counts.get((prev_word, current_word))
                if trigram_candidates:
                    result["trigram"] = self._format_probabilities(trigram_candidates)
        
        return result

    def _format_probabilities(self, candidates):
        """Olasılıkları formatlar ve sıralar"""
        total = sum(candidates.values())
        probabilities = [
            {
                "word": word,
                "count": count,
                "probability":  round((count / total) * 100, 1)
            }
            for word, count in candidates.items()
        ]
        # En yüksek olasılıktan düşüğe sırala
        probabilities.sort(key=lambda x: x["probability"], reverse=True)
        return probabilities[:5]  # En fazla 5 kelime göster

    def predict(self, full_text):
        """
        Kelime tamamlama + sonraki kelime önerisi
        """
        if not full_text or full_text. endswith(" "):
            return ""

        words_list = full_text.split()
        current_partial = words_list[-1].lower()
        
        matches = [w for w in self.word_counts if w.startswith(current_partial)]
        if not matches:
            return ""

        best_completion = max(matches, key=lambda w: self.word_counts[w])
        suffix = best_completion[len(current_partial):]
        
        # TRIGRAM ile sonraki kelimeyi seç
        if len(words_list) >= 2:
            prev_word = words_list[-2].lower()
            current_word = best_completion
            
            next_word_candidates = self.trigram_counts.get((prev_word, current_word))
            
            if next_word_candidates: 
                words = list(next_word_candidates.keys())
                counts = list(next_word_candidates.values())
                chosen_next_word = random.choices(words, weights=counts, k=1)[0]
                return suffix + " " + chosen_next_word
        
        # Trigram bulunamazsa bigram'a düş
        next_word_candidates = self.bigram_counts[best_completion]
        
        if next_word_candidates: 
            words = list(next_word_candidates.keys())
            counts = list(next_word_candidates.values())
            chosen_next_word = random.choices(words, weights=counts, k=1)[0]
            return suffix + " " + chosen_next_word
        
        return suffix

# Modeli Başlat
ai = TrigramPredictor("story.txt")

# HTML kodu
HTML_CODE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>AI 2-gram & 3-gram Writer</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            min-height: 100vh; 
            background:  #202124; 
            color: white; 
            margin: 0;
            padding: 20px;
            box-sizing: border-box;
        }
        .main-container {
            width: 900px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .container { position: relative; width: 100%; }

        . layer {
            font-family:  'Courier New', Courier, monospace;
            font-size: 24px;
            width: 100%;
            padding:  20px;
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

        . hint { 
            margin-top: 10px; 
            text-align: center; 
            color: #9aa0a6; 
            font-size:  0.9em; 
        }
        kbd { 
            background: #3c4043; 
            padding:  4px 8px; 
            border-radius: 4px; 
            border: 1px solid #5f6368; 
        }

        /* Olasılık Kutuları Grid */
        .probabilities-grid {
            display:  grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        /* Olasılık Kutusu Stilleri */
        .probability-box {
            background: rgba(60, 64, 67, 0.9);
            border: 1px solid #5f6368;
            border-radius:  8px;
            padding:  15px;
            animation: slideIn 0.3s ease;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform:  translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .probability-box.hidden {
            display: none;
        }

        .probability-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom:  15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #5f6368;
        }

        .gram-badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 11px;
            font-weight: 700;
            padding: 4px 10px;
            border-radius:  12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .gram-badge.bigram {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }

        .gram-badge.trigram {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }

        .probability-title {
            font-size: 13px;
            color: #e8eaed;
            font-weight: 600;
            flex:  1;
        }

        .current-word {
            color: #8ab4f8;
            font-family: 'Courier New', Courier, monospace;
            font-weight: 700;
        }

        .probability-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .probability-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 12px;
            background: rgba(138, 180, 248, 0.1);
            border-radius: 6px;
            transition: all 0.2s;
        }

        .probability-item:hover {
            background:  rgba(138, 180, 248, 0.2);
            transform: translateX(5px);
        }

        .probability-item.top-choice {
            background: rgba(138, 180, 248, 0.2);
            border-left: 3px solid #8ab4f8;
        }

        .word-info {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        . word-text {
            font-family: 'Courier New', Courier, monospace;
            font-size: 15px;
            color: #e8eaed;
            font-weight: 600;
        }

        .word-rank {
            background: #5f6368;
            color:  #9aa0a6;
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight:  600;
        }

        .probability-bar-container {
            display: flex;
            align-items: center;
            gap: 8px;
            flex:  1;
            max-width: 150px;
        }

        . probability-bar {
            height:  6px;
            background: rgba(95, 99, 104, 0.3);
            border-radius: 3px;
            flex:  1;
            overflow: hidden;
        }

        .probability-fill {
            height: 100%;
            background: linear-gradient(90deg, #8ab4f8, #4285f4);
            border-radius: 3px;
            transition: width 0.3s ease;
        }

        .probability-percent {
            font-size: 12px;
            color: #8ab4f8;
            font-weight: 600;
            min-width: 42px;
            text-align: right;
        }

        .no-predictions {
            color: #9aa0a6;
            font-style: italic;
            text-align: center;
            padding: 15px;
            font-size: 13px;
        }

        . info-banner {
            background: rgba(138, 180, 248, 0.1);
            border: 1px solid rgba(138, 180, 248, 0.3);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
            color: #9aa0a6;
            font-size: 13px;
        }
    </style>
</head>
<body>
    <div class="main-container">
        <div class="container">
            <div id="ghost-input" class="layer"></div>
            <textarea id="user-input" class="layer" placeholder="Yazmaya başla..." autofocus></textarea>
            <div class="hint">Tamamlama için <kbd>TAB</kbd> | 2-gram ve 3-gram olasılıkları aşağıda</div>
        </div>

        <div id="probabilities-container" class="probabilities-grid" style="display: none;">
            <!-- 2-GRAM (Bigram) -->
            <div id="bigram-box" class="probability-box">
                <div class="probability-header">
                    <span class="gram-badge bigram">2-GRAM</span>
                    <span class="probability-title">"<span id="bigram-word" class="current-word"></span>" sonrası</span>
                </div>
                <div id="bigram-list" class="probability-list">
                    <div class="no-predictions">Veri yok</div>
                </div>
            </div>

            <!-- 3-GRAM (Trigram) -->
            <div id="trigram-box" class="probability-box">
                <div class="probability-header">
                    <span class="gram-badge trigram">3-GRAM</span>
                    <span class="probability-title">"<span id="trigram-context" class="current-word"></span>" sonrası</span>
                </div>
                <div id="trigram-list" class="probability-list">
                    <div class="no-predictions">Veri yok</div>
                </div>
            </div>
        </div>

        <div id="info-banner" class="info-banner">
            💡 2-gram:  Son 1 kelimeye bakar | 3-gram: Son 2 kelimeye bakar (daha akıllı!)
        </div>
    </div>

    <script>
        const inputField = document.getElementById('user-input');
        const ghostField = document.getElementById('ghost-input');
        const probabilitiesContainer = document.getElementById('probabilities-container');
        
        const bigramBox = document.getElementById('bigram-box');
        const trigramBox = document.getElementById('trigram-box');
        
        const bigramList = document.getElementById('bigram-list');
        const trigramList = document.getElementById('trigram-list');
        
        const bigramWordSpan = document.getElementById('bigram-word');
        const trigramContextSpan = document.getElementById('trigram-context');
        
        let currentSuggestion = "";

        inputField.addEventListener('input', async function() {
            const text = this. value;
            
            // Tahmin için API çağrısı (tamamlama)
            if (text.length > 0 && ! text.endsWith(" ")) {
                const response = await fetch('/predict? text=' + encodeURIComponent(text));
                const data = await response.json();
                currentSuggestion = data.prediction;

                if (currentSuggestion) {
                    ghostField.textContent = text + currentSuggestion;
                } else {
                    ghostField. textContent = "";
                }
            } else {
                ghostField.textContent = "";
                currentSuggestion = "";
            }

            // Olasılıklar için API çağrısı
            if (text.length > 0) {
                const probResponse = await fetch('/probabilities?text=' + encodeURIComponent(text));
                const probData = await probResponse.json();

                updateProbabilities(probData, text);
            } else {
                probabilitiesContainer.style.display = 'none';
            }
        });

        function updateProbabilities(data, text) {
            const hasBigram = data.bigram && data.bigram.length > 0;
            const hasTrigram = data.trigram && data.trigram.length > 0;

            if (hasBigram || hasTrigram) {
                probabilitiesContainer.style.display = 'grid';
                
                // 2-GRAM güncellemesi
                if (hasBigram) {
                    bigramWordSpan.textContent = data. current_word;
                    renderProbabilities(bigramList, data.bigram);
                    bigramBox.classList.remove('hidden');
                } else {
                    bigramList.innerHTML = '<div class="no-predictions">Veri yok</div>';
                }

                // 3-GRAM güncellemesi
                if (hasTrigram) {
                    const words = text.split(' ');
                    let context = '';
                    if (text.endsWith(' ')) {
                        context = words. length >= 2 ? words[words. length - 2] + ' ' + words[words.length - 1] : words[words.length - 1];
                    } else {
                        context = words. length >= 2 ? words[words.length - 2] + ' ' + words[words.length - 1] : words[words.length - 1];
                    }
                    trigramContextSpan.textContent = context;
                    renderProbabilities(trigramList, data.trigram);
                    trigramBox. classList.remove('hidden');
                } else {
                    trigramList.innerHTML = '<div class="no-predictions">En az 2 kelime gerekli</div>';
                }
            } else {
                probabilitiesContainer.style.display = 'none';
            }
        }

        function renderProbabilities(container, probabilities) {
            container.innerHTML = '';
            
            probabilities. forEach((item, index) => {
                const itemDiv = document.createElement('div');
                itemDiv.className = 'probability-item' + (index === 0 ?  ' top-choice' : '');
                
                itemDiv.innerHTML = `
                    <div class="word-info">
                        <span class="word-rank">#${index + 1}</span>
                        <span class="word-text">${item.word}</span>
                    </div>
                    <div class="probability-bar-container">
                        <div class="probability-bar">
                            <div class="probability-fill" style="width: ${item.probability}%"></div>
                        </div>
                        <span class="probability-percent">${item.probability}%</span>
                    </div>
                `;
                
                container.appendChild(itemDiv);
            });
        }

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

@app.route('/probabilities')
def get_probabilities():
    text = request.args.get('text', '')
    probabilities = ai. get_probabilities(text)
    return jsonify(probabilities)

if __name__ == '__main__':
    app.run(debug=True)