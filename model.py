"""
N-gram + LSTM Hybrid Tahmin Modeli
"""
from collections import Counter, defaultdict
import os
import pickle
import random
from text_utils import (
    full_clean, is_sentence_ending, is_punctuation_token,
    restore_punctuation_from_tokens
)

class NgramPredictor:
    """N-gram + LSTM hybrid tahmin modeli."""
    
    def __init__(self, load_from_pickle=None, lstm_model_path=None):
        """
        Modeli başlat.
        Eğer load_from_pickle verilirse dosyadan yükler, verilmezse boş başlar (eğitim için).
        lstm_model_path: LSTM model dosyası (opsiyonel)
        """
        self.word_counts = Counter()
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.trigram_counts = defaultdict(Counter)
        
        self.total_words = 0
        
        # LSTM model (opsiyonel)
        self.lstm_model = None
        if lstm_model_path and os.path.exists(lstm_model_path):
            try:
                from transformer_model import TransformerPredictor
                self.lstm_model = TransformerPredictor(lstm_model_path)
                print("✅ LSTM model yüklendi (hybrid mod aktif)")
            except Exception as e:
                print(f"⚠️ LSTM model yüklenemedi: {e}")
        
        if load_from_pickle:
            if os.path.exists(load_from_pickle):
                self.load_from_pickle(load_from_pickle)
            else:
                print(f"⚠️ Uyarı: {load_from_pickle} bulunamadı, boş model başlatıldı.")

    def train_from_file(self, filepath):
        """Verilen dosyadan modeli eğitir."""
        print(f"📚 Eğitim başlıyor: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Kelimeleri ayır (zaten tokenized olduğu için split yeterli)
        tokens = text.split()
        self.total_words = len(tokens)
        
        print("   Tokenlar sayılıyor...")
        
        # 1. Unigram & Kelime Sayımları
        self.word_counts.update(tokens)
        self.unigram_counts.update(tokens)
        
        # 2. Bigram (İkili)
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i+1]
            self.bigram_counts[w1][w2] += 1
            
        # 3. Trigram (Üçlü)
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
            # (kelime1, kelime2) -> kelime3
            self.trigram_counts[(w1, w2)][w3] += 1
            
        print(f"✅ Eğitim tamamlandı!")
        print(f"   - {len(self.unigram_counts)} benzersiz kelime")
        print(f"   - {len(self.bigram_counts)} bigram kuralı")
        print(f"   - {len(self.trigram_counts)} trigram kuralı")

    def load_from_pickle(self, filepath):
        """Pickle dosyasından yükle."""
        print(f"📦 Model yükleniyor: {filepath}")
        with open(filepath, 'rb') as f:
            loaded = pickle.load(f)
            
        # Dictionary formatındaysa
        if isinstance(loaded, dict):
            self.word_counts = loaded.get('word_counts', Counter())
            self.unigram_counts = loaded.get('unigram_counts', Counter())
            self.bigram_counts = defaultdict(Counter, loaded.get('bigram_counts', {}))
            self.trigram_counts = defaultdict(Counter, loaded.get('trigram_counts', {}))
            self.total_words = loaded.get('total_words', 0)
        # Sınıf örneği ise (__dict__ transferi)
        elif hasattr(loaded, '__dict__'):
            self.__dict__.update(loaded.__dict__)
            
        print(f"✅ Model yüklendi. Kelime haznesi: {len(self.word_counts):,}")

    def predict(self, text, use_tokens=True):
        """Kelime tamamlama."""
        if not text or text.endswith(" "):
            return ""
        
        text_clean = full_clean(text, lowercase=True, use_tokens=use_tokens)
        words = text_clean.split()
        if not words: return ""
        
        current = words[-1]
        
        # Noktalama tokeni tamamlanmaz
        if is_punctuation_token(current): return ""
        
        # Kelimeyi tamamla
        matches = [w for w in self.word_counts if w.startswith(current)]
        if not matches: return ""
        
        # En sık kullanılanı seç
        best = max(matches, key=lambda w: self.word_counts[w])
        suffix = best[len(current):]
        
        # Bir sonraki kelimeyi tahmin et (Bigram)
        next_word_pred = ""
        if best in self.bigram_counts:
            candidates = self.bigram_counts[best]
            if candidates:
                # Ağırlıklı rastgele seçim
                next_word = random.choices(list(candidates.keys()), weights=list(candidates.values()), k=1)[0]
                next_word_pred = " " + next_word
        
        result = suffix + next_word_pred
        
        if use_tokens:
            result = restore_punctuation_from_tokens(result)
            
        return result

    def get_probabilities(self, text, use_tokens=True):
        """Olasılıkları hesapla (Front-end için)."""
        # Basit versiyon: Sadece bigram ve trigram döndürür
        text_clean = full_clean(text, lowercase=True, use_tokens=use_tokens)
        words = text_clean.split()
        
        if not words: return {"unigram": [], "bigram": [], "trigram": []}
        
        last_word = words[-1]
        
        # Bigram Olasılıkları
        bigram_probs = []
        if last_word in self.bigram_counts:
            total = sum(self.bigram_counts[last_word].values())
            for word, count in self.bigram_counts[last_word].most_common(5):
                # Token ise restore et
                display_word = restore_punctuation_from_tokens(word) if is_punctuation_token(word) else word
                bigram_probs.append({
                    "word": display_word, 
                    "probability": round((count/total)*100, 1)
                })

        # Trigram Olasılıkları
        trigram_probs = []
        if len(words) >= 2:
            prev_word = words[-2]
            key = (prev_word, last_word)
            if key in self.trigram_counts:
                total = sum(self.trigram_counts[key].values())
                for word, count in self.trigram_counts[key].most_common(5):
                    # Token ise restore et
                    display_word = restore_punctuation_from_tokens(word) if is_punctuation_token(word) else word
                    trigram_probs.append({
                        "word": display_word, 
                        "probability": round((count/total)*100, 1)
                    })
        
        # Current word'u de restore et
        display_current = restore_punctuation_from_tokens(last_word) if is_punctuation_token(last_word) else last_word
                    
        return {
            "unigram": [], # Şimdilik boş geçiyoruz
            "bigram": bigram_probs,
            "trigram": trigram_probs,
            "current_word": display_current
        }

    def predict_until_sentence_end(self, text, use_tokens=True, max_words=50):
        """
        Bir sonraki cümle bitirici noktalama işaretine kadar devam eder.
        Cümle tamamlama için kullanılır (Shift+Tab).
        
        Args:
            text: Başlangıç metni
            use_tokens: Noktalama tokenları kullan
            max_words: Maksimum kelime sayısı (sonsuz döngüyü önlemek için)
            
        Returns:
            Tamamlanmış cümle parçası
        """
        # Metni temizle
        text_clean = full_clean(text, lowercase=True, use_tokens=use_tokens)
        words = text_clean.split()
        
        if not words:
            return ""
        
        generated_words = []
        last_word = words[-1]
        
        # Eğer son kelime noktalama tokeni değilse, önce kelimeyi tamamla
        if not is_punctuation_token(last_word) and not text.endswith(" "):
            # Kelimeyi tamamlama
            matches = [w for w in self.word_counts if w.startswith(last_word)]
            if matches:
                # En sık kullanılanı seç
                best = max(matches, key=lambda w: self.word_counts[w])
                suffix = best[len(last_word):]
                if suffix:
                    generated_words.append(suffix)
                # Context'i güncelle
                words[-1] = best
        
        # Şimdi cümleyi devam ettir
        current_context = words[-2:] if len(words) >= 2 else words[-1:]
        
        # Cümle bitirici token bulunana kadar devam et
        for _ in range(max_words):
            # Trigram varsa kullan
            if len(current_context) >= 2:
                key = tuple(current_context[-2:])
                if key in self.trigram_counts:
                    candidates = self.trigram_counts[key]
                    if candidates:
                        # Ağırlıklı rastgele seçim
                        next_word = random.choices(
                            list(candidates.keys()), 
                            weights=list(candidates.values()), 
                            k=1
                        )[0]
                        generated_words.append(next_word)
                        current_context.append(next_word)
                        
                        # Cümle bitirici token kontrolü
                        if is_sentence_ending(next_word):
                            break
                        continue
            
            # Bigram varsa kullan
            if len(current_context) >= 1:
                last_word = current_context[-1]
                if last_word in self.bigram_counts:
                    candidates = self.bigram_counts[last_word]
                    if candidates:
                        next_word = random.choices(
                            list(candidates.keys()), 
                            weights=list(candidates.values()), 
                            k=1
                        )[0]
                        generated_words.append(next_word)
                        current_context.append(next_word)
                        
                        # Cümle bitirici token kontrolü
                        if is_sentence_ending(next_word):
                            break
                        continue
            
            # Hiçbir kandidat yoksa dur
            break
        
        # Sonucu oluştur
        result = " ".join(generated_words)
        
        # Tokenları restore et
        if use_tokens:
            result = restore_punctuation_from_tokens(result)
        
        return result