from collections import Counter, defaultdict
import os
import re
import random

# Token tanımları (preprocess_story.py ile aynı)
PUNCTUATION_TOKENS = {
    '.': 'TR001',
    ',': 'TR002',
    '!': 'TR003',
    '?': 'TR004',
    ';': 'TR005',
    ':': 'TR006',
    '-': 'TR007',
    '—': 'TR008',
    '–': 'TR009',
    '(': 'TR010',
    ')': 'TR011',
    '"': 'TR012',
    "'": 'TR013',
    '«': 'TR014',
    '»': 'TR015',
    '...': 'TR016',
    '\n': 'TR017',
    '\t': 'TR018',
}

TOKEN_TO_PUNCT = {v: k for k, v in PUNCTUATION_TOKENS.items()}

# Cümle bitiren tokenler
SENTENCE_ENDERS = ['TR001', 'TR003', 'TR004']  # . ! ?


class NgramPredictor:
    """
    Sentence-based n-gram model - noktalama işaretleri ve sayıları korur
    """
    
    def __init__(self, filename):
        self.word_counts = Counter()
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.trigram_counts = defaultdict(Counter)
        self.total_words = 0
        self.load_data(filename)

    def load_data(self, filename):
        """Tokenize edilmiş metin dosyasını yükler ve n-gram modellerini oluşturur."""
        if not os.path.exists(filename):
            print(f"HATA: {filename} bulunamadı.")
            return
            
        with open(filename, "r", encoding="utf-8") as file:
            text = file.read()
        
        # Metni küçük harfe çevir ama token'ları koru
        words = text.split()
        processed_words = []
        
        for word in words:
            # Token ise olduğu gibi bırak, değilse küçük harfe çevir
            if word.startswith('TR') and word[2:5].isdigit():
                processed_words.append(word)
            else:
                processed_words.append(word.lower())
        
        words = processed_words
        
        # Kelime frekansları
        self.word_counts = Counter(words)
        self.total_words = len(words)
        
        # 1-gram (unigram)
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
        
        print(f"✓ Model yüklendi (Sentence-based)")
        print(f"  - Benzersiz token/kelime: {len(self.word_counts)}")
        print(f"  - Toplam token:  {self.total_words}")
        print(f"  - 2-gram sayısı: {len(self.bigram_counts)}")
        print(f"  - 3-gram sayısı: {len(self.trigram_counts)}")

    def detokenize(self, text):
        """Token'ları noktalama işaretlerine çevirir."""
        for token, punct in TOKEN_TO_PUNCT.items():
            text = text.replace(token, punct)
        
        # Noktalama düzeltmeleri
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])', r'\1 ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    def get_probabilities(self, full_text):
        """
        Sentence-based olasılık hesaplama
        """
        if not full_text:
            unigram_probs = self._format_probabilities(self.unigram_counts, use_total=True)
            # Unigram sonuçlarını detokenize et
            for prob in unigram_probs:
                prob['word'] = self.detokenize(prob['word'])
            return {
                "unigram": unigram_probs,
                "bigram": [],
                "trigram": [],
                "current_word": "",
                "context": ""
            }

        # Metni tokenize et (kullanıcı input'u)
        words_list = full_text.split()
        
        if full_text.endswith(" "):
            if len(words_list) < 1:
                unigram_probs = self._format_probabilities(self.unigram_counts, use_total=True)
                for prob in unigram_probs:
                    prob['word'] = self.detokenize(prob['word'])
                return {
                    "unigram": unigram_probs,
                    "bigram": [],
                    "trigram": [],
                    "current_word": "",
                    "context": ""
                }
            current_word = words_list[-1].lower()
        else:
            current_word = words_list[-1].lower()
        
        result = {
            "unigram": [],
            "bigram": [],
            "trigram": [],
            "current_word": self.detokenize(current_word),
            "context": ""
        }
        
        # 1-GRAM olasılıkları
        unigram_probs = self._format_probabilities(self.unigram_counts, use_total=True)
        for prob in unigram_probs:
            prob['word'] = self.detokenize(prob['word'])
        result["unigram"] = unigram_probs
        
        # 2-GRAM olasılıkları
        bigram_candidates = self.bigram_counts.get(current_word)
        if bigram_candidates:
            bigram_probs = self._format_probabilities(bigram_candidates)
            for prob in bigram_probs:
                prob['word'] = self.detokenize(prob['word'])
            result["bigram"] = bigram_probs
        
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
                result["context"] = self.detokenize(f"{prev_word} {current_word}")
                trigram_candidates = self.trigram_counts.get((prev_word, current_word))
                if trigram_candidates:
                    trigram_probs = self._format_probabilities(trigram_candidates)
                    for prob in trigram_probs:
                        prob['word'] = self.detokenize(prob['word'])
                    result["trigram"] = trigram_probs
        
        return result

    def _format_probabilities(self, candidates, use_total=False):
        """Olasılıkları yüzde olarak formatlar ve sıralar."""
        if use_total:
            total = self.total_words
        else:
            total = sum(candidates.values())
        
        if total == 0:
            return []
            
        probabilities = [
            {
                "word": word,
                "count": count,
                "probability": round((count / total) * 100, 1)
            }
            for word, count in candidates.items()
        ]
        probabilities.sort(key=lambda x: x["probability"], reverse=True)
        return probabilities[:10]  # En fazla 10 sonuç

    def predict(self, full_text):
        """
        Sentence-based kelime tamamlama ve devam ettirme.
        Cümle sonu karakteri geldiğinde yeni cümle başlatır.
        """
        if not full_text:
            # Rastgele bir başlangıç kelimesi seç
            common_words = [w for w, c in self.word_counts.most_common(50) if not w.startswith('TR')]
            if common_words:
                return random.choice(common_words)
            return ""

        if full_text.endswith(" "):
            # Sonraki kelimeyi tahmin et
            words_list = full_text.strip().split()
            
            if len(words_list) == 0:
                common_words = [w for w, c in self.word_counts.most_common(50) if not w.startswith('TR')]
                return random.choice(common_words) if common_words else ""
            
            last_word = words_list[-1].lower()
            
            # Son kelime cümle bitirici mi kontrol et
            is_sentence_ender = any(last_word.endswith(self.detokenize(token)) for token in SENTENCE_ENDERS)
            
            # 3-gram dene
            if len(words_list) >= 2:
                prev_word = words_list[-2].lower()
                next_word_candidates = self.trigram_counts.get((prev_word, last_word))
                
                if next_word_candidates:
                    words = list(next_word_candidates.keys())
                    counts = list(next_word_candidates.values())
                    chosen_word = random.choices(words, weights=counts, k=1)[0]
                    return self.detokenize(chosen_word)
            
            # 2-gram dene
            next_word_candidates = self.bigram_counts.get(last_word)
            if next_word_candidates:
                words = list(next_word_candidates.keys())
                counts = list(next_word_candidates.values())
                chosen_word = random.choices(words, weights=counts, k=1)[0]
                return self.detokenize(chosen_word)
            
            # Fallback: rastgele yaygın kelime
            common_words = [w for w, c in self.word_counts.most_common(30) if not w.startswith('TR')]
            return random.choice(common_words) if common_words else ""
        
        else:
            # Kelime tamamlama
            words_list = full_text.split()
            current_partial = words_list[-1].lower()
            
            # Mevcut kelimeyi tamamla
            matches = [w for w in self.word_counts if w.startswith(current_partial) and not w.startswith('TR')]
            
            if not matches:
                return ""
            
            best_completion = max(matches, key=lambda w: self.word_counts[w])
            suffix = best_completion[len(current_partial):]
            
            # Sonraki kelimeyi de ekle
            if len(words_list) >= 2:
                prev_word = words_list[-2].lower()
                next_word_candidates = self.trigram_counts.get((prev_word, best_completion))
                
                if next_word_candidates:
                    words = list(next_word_candidates.keys())
                    counts = list(next_word_candidates.values())
                    chosen_next = random.choices(words, weights=counts, k=1)[0]
                    return suffix + " " + self.detokenize(chosen_next)
            
            # 2-gram ile devam et
            next_word_candidates = self.bigram_counts.get(best_completion)
            if next_word_candidates:
                words = list(next_word_candidates.keys())
                counts = list(next_word_candidates.values())
                chosen_next = random.choices(words, weights=counts, k=1)[0]
                return suffix + " " + self.detokenize(chosen_next)
            
            return suffix
