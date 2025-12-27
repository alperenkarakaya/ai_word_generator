from collections import Counter, defaultdict
import os
import re
import random


class NgramPredictor:
    """
    1-gram, 2-gram ve 3-gram tabanlı kelime tahmin modeli. 
    """
    
    def __init__(self, filename):
        self.word_counts = Counter()
        self.unigram_counts = Counter()  # 1-gram (kelime frekansları)
        self.bigram_counts = defaultdict(Counter)
        self.trigram_counts = defaultdict(Counter)
        self.total_words = 0
        self.load_data(filename)

    def load_data(self, filename):
        """Metin dosyasını yükler ve n-gram modellerini oluşturur."""
        if not os.path.exists(filename):
            print(f"HATA: {filename} bulunamadı.")
            return
            
        with open(filename, "r", encoding="utf-8") as file:
            text = file.read().lower()
        
        # Metni temizle
        text_clean = re.sub(r'[^\w\s]', '', text)
        words = text_clean.split()
        
        # Kelime frekansları
        self.word_counts = Counter(words)
        self.total_words = len(words)
        
        # 1-gram (unigram) - Genel kelime olasılıkları
        self.unigram_counts = self.word_counts.copy()
        
        # 2-gram (bigram) oluştur
        for i in range(len(words) - 1):
            current_w = words[i]
            next_w = words[i + 1]
            self.bigram_counts[current_w][next_w] += 1
        
        # 3-gram (trigram) oluştur
        for i in range(len(words) - 2):
            word1 = words[i]
            word2 = words[i + 1]
            next_word = words[i + 2]
            self.trigram_counts[(word1, word2)][next_word] += 1
        
        print(f"✓ Model yüklendi")
        print(f"  - Benzersiz kelime:  {len(self.word_counts)}")
        print(f"  - Toplam kelime:  {self.total_words}")
        print(f"  - 2-gram sayısı: {len(self. bigram_counts)}")
        print(f"  - 3-gram sayısı: {len(self.trigram_counts)}")

    def get_probabilities(self, full_text):
        """
        Verilen metin için 1-gram, 2-gram ve 3-gram olasılıklarını hesaplar.
        """
        if not full_text:  
            # Boş metin - sadece 1-gram göster
            unigram_probs = self._format_probabilities(self.unigram_counts, use_total=True)
            return {
                "unigram": unigram_probs,
                "bigram":  [],
                "trigram": [],
                "current_word": "",
                "context":  ""
            }

        words_list = full_text.split()
        
        # Şu anki kelimeyi belirle (TAMAMLAMA YAPMA!)
        if full_text.endswith(" "):
            if len(words_list) < 1:
                unigram_probs = self._format_probabilities(self.unigram_counts, use_total=True)
                return {
                    "unigram": unigram_probs,
                    "bigram": [],
                    "trigram": [],
                    "current_word": "",
                    "context": ""
                }
            current_word = words_list[-1]. lower()
        else:
            current_word = words_list[-1].lower()
        
        result = {
            "unigram": [],
            "bigram": [],
            "trigram": [],
            "current_word": current_word,
            "context":  ""
        }
        
        # 1-GRAM (UNIGRAM) olasılıkları - Genel kelime frekansları
        result["unigram"] = self._format_probabilities(self.unigram_counts, use_total=True)
        
        # 2-GRAM olasılıkları
        bigram_candidates = self.bigram_counts. get(current_word)
        if bigram_candidates: 
            result["bigram"] = self._format_probabilities(bigram_candidates)
        
        # 3-GRAM olasılıkları
        if len(words_list) >= 2:
            if full_text.endswith(" "):
                prev_word = words_list[-2].lower()
            else:
                if len(words_list) >= 2:
                    prev_word = words_list[-2].lower()
                else:
                    prev_word = None
            
            if prev_word:
                result["context"] = f"{prev_word} {current_word}"
                trigram_candidates = self.trigram_counts.get((prev_word, current_word))
                if trigram_candidates:
                    result["trigram"] = self._format_probabilities(trigram_candidates)
        
        return result

    def _format_probabilities(self, candidates, use_total=False):
        """Olasılıkları yüzde olarak formatlar ve sıralar."""
        if use_total:
            # 1-gram için toplam kelime sayısını kullan
            total = self.total_words
        else:
            total = sum(candidates.values())
        
        if total == 0:
            return []
            
        probabilities = [
            {
                "word": word,
                "count": count,
                "probability":  round((count / total) * 100, 1)
            }
            for word, count in candidates.items()
        ]
        probabilities.sort(key=lambda x: x["probability"], reverse=True)
        return probabilities[:5]  # En fazla 5 kelime

    def predict(self, full_text):
        """
        Kelime tamamlama + sonraki kelime önerisi. 
        """
        if not full_text or full_text.endswith(" "):
            return ""

        words_list = full_text.split()
        current_partial = words_list[-1].lower()
        
        # Kelime tamamlama
        matches = [w for w in self.word_counts if w.startswith(current_partial)]
        if not matches:
            return ""

        best_completion = max(matches, key=lambda w: self.word_counts[w])
        suffix = best_completion[len(current_partial):]
        
        # 3-gram ile sonraki kelimeyi tahmin et
        if len(words_list) >= 2:
            prev_word = words_list[-2].lower()
            current_word = best_completion
            
            next_word_candidates = self.trigram_counts.get((prev_word, current_word))
            
            if next_word_candidates: 
                words = list(next_word_candidates.keys())
                counts = list(next_word_candidates.values())
                chosen_next_word = random.choices(words, weights=counts, k=1)[0]
                return suffix + " " + chosen_next_word
        
        # 2-gram'a düş
        next_word_candidates = self.bigram_counts[best_completion]
        
        if next_word_candidates: 
            words = list(next_word_candidates.keys())
            counts = list(next_word_candidates.values())
            chosen_next_word = random.choices(words, weights=counts, k=1)[0]
            return suffix + " " + chosen_next_word
        
        return suffix