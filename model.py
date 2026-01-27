from collections import Counter, defaultdict
import os
import re
import random
import pickle

# Torch'u opsiyonel yap
try:
    import torch
    from transformer_model import TurkishGPT, TransformerConfig
    from tokenizer import SimpleTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️  PyTorch bulunamadı, sadece N-gram kullanılacak")
    TORCH_AVAILABLE = False

from text_utils import (
    full_clean, tokenize_with_sentences, get_sentence_context,
    is_sentence_ending, strip_punctuation
)


class NgramPredictor:
    """
    1-gram, 2-gram ve 3-gram tabanlı kelime tahmin modeli.
    Cümle bağlamı desteği ve Transformer entegrasyonu ile geliştirilmiş versiyon.
    """
    
    def __init__(self, filename=None, load_from_pickle=None):
        """
        Model başlatıcı.
        
        Args:
            filename: story.txt gibi metin dosyası (ilk eğitim için)
            load_from_pickle: saved_model.pkl gibi pickle dosyası (hızlı yükleme)
        """
        self.word_counts = Counter()
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.trigram_counts = defaultdict(Counter)
        
        # Cümle bağlamı için (sadece .!? sonrası)
        self.cross_sentence_bigrams = defaultdict(Counter)
        self.cross_sentence_trigrams = defaultdict(Counter)
        
        self.total_words = 0
        self.tokens = []
        self.sentence_ids = []
        
        # Transformer modeli (lazy loading)
        self.transformer_model = None
        self.transformer_tokenizer = None
        
        if TORCH_AVAILABLE:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = None
        
        # Pickle'dan yükle VEYA metin dosyasından eğit
        if load_from_pickle and os.path.exists(load_from_pickle):
            self.load_from_pickle(load_from_pickle)
        elif filename:
            self.load_data(filename)
        else:
            raise ValueError("filename veya load_from_pickle parametresi gerekli!")
        
        # Transformer model varsa yükle
        if TORCH_AVAILABLE:
            self._load_transformer()

    def load_data(self, filename):
        """Metin dosyasını yükler ve n-gram modellerini oluşturur."""
        print(f"📖 Metin dosyası okunuyor: {filename}")
        
        if not os.path.exists(filename):
            print(f"HATA: {filename} bulunamadı.")
            return
            
        with open(filename, "r", encoding="utf-8") as file:
            text = file.read()
        
        # Metni tam temizle (text_utils kullanarak)
        text_clean = full_clean(text, lowercase=True, keep_punctuation=True)
        
        # Tokenize et ve cümle ID'lerini al
        self.tokens, self.sentence_ids = tokenize_with_sentences(text_clean)
        
        # Noktalama işaretlerini temizle (n-gram için)
        words = [strip_punctuation(token) for token in self.tokens]
        
        # Kelime frekansları
        self.word_counts = Counter(words)
        self.total_words = len(words)
        
        # 1-gram (unigram) - Genel kelime olasılıkları
        self.unigram_counts = self.word_counts.copy()
        
        # 2-gram ve 3-gram oluştur
        for i in range(len(self.tokens) - 1):
            current_token = self.tokens[i]
            next_token = self.tokens[i + 1]
            
            current_word = strip_punctuation(current_token)
            next_word = strip_punctuation(next_token)
            
            # Normal bigram
            self.bigram_counts[current_word][next_word] += 1
            
            # Cross-sentence bigram (sadece .!? sonrası)
            if is_sentence_ending(current_token):
                self.cross_sentence_bigrams[current_word][next_word] += 1
        
        # 3-gram (trigram) oluştur
        for i in range(len(self.tokens) - 2):
            token1 = self.tokens[i]
            token2 = self.tokens[i + 1]
            token3 = self.tokens[i + 2]
            
            word1 = strip_punctuation(token1)
            word2 = strip_punctuation(token2)
            word3 = strip_punctuation(token3)
            
            # Normal trigram
            self.trigram_counts[(word1, word2)][word3] += 1
            
            # Cross-sentence trigram (token2 cümle bitirici ise)
            if is_sentence_ending(token2):
                self.cross_sentence_trigrams[(word1, word2)][word3] += 1
        
        print(f"✓ Model eğitildi")
        print(f"  - Benzersiz kelime: {len(self.word_counts)}")
        print(f"  - Toplam kelime: {self.total_words}")
        print(f"  - 2-gram sayısı: {len(self.bigram_counts)}")
        print(f"  - 3-gram sayısı: {len(self.trigram_counts)}")
        print(f"  - Cross-sentence bigram: {len(self.cross_sentence_bigrams)}")
        print(f"  - Cross-sentence trigram: {len(self.cross_sentence_trigrams)}")

    def save_to_pickle(self, filepath='saved_model.pkl'):
        """
        Modeli pickle dosyasına kaydet.
        
        Args:
            filepath: Kaydedilecek dosya yolu
        """
        print(f"💾 Model kaydediliyor: {filepath}")
        
        model_data = {
            'word_counts': self.word_counts,
            'unigram_counts': self.unigram_counts,
            'bigram_counts': dict(self.bigram_counts),
            'trigram_counts': dict(self.trigram_counts),
            'cross_sentence_bigrams': dict(self.cross_sentence_bigrams),
            'cross_sentence_trigrams': dict(self.cross_sentence_trigrams),
            'total_words': self.total_words,
            'tokens': self.tokens,
            'sentence_ids': self.sentence_ids,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"✅ Model kaydedildi: {filepath}")

    def load_from_pickle(self, filepath='saved_model.pkl'):
        """
        Pickle dosyasından modeli yükle.
        Hem eski format (tüm nesne) hem yeni format (dict) desteklenir.
        
        Args:
            filepath: Yüklenecek dosya yolu
        """
        print(f"📦 Model yükleniyor: {filepath}")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"❌ Pickle dosyası bulunamadı: {filepath}")

        with open(filepath, 'rb') as f:
            loaded_data = pickle.load(f)

        # ESKİ FORMAT: Tüm NgramPredictor nesnesi kaydedilmiş
        # veya başka bir modülde aynı alanlara sahip bir nesne (ör. model_new.NgramPredictor)
        if isinstance(loaded_data, NgramPredictor) or (
            not isinstance(loaded_data, dict) and
            hasattr(loaded_data, 'word_counts') and
            hasattr(loaded_data, 'unigram_counts')
        ):
            print("🔄 Eski format tespit edildi, veriler aktarılıyor...")

            try:
                # Daha toleranslı yükleme: eksik alanlarda varsayılan atama yap
                def _get(obj, name, default=None):
                    if hasattr(obj, name):
                        return getattr(obj, name)
                    if isinstance(obj, dict) and name in obj:
                        return obj[name]
                    if hasattr(obj, '__dict__') and name in obj.__dict__:
                        return obj.__dict__[name]
                    return default

                # Kelime sayıları
                wc = _get(loaded_data, 'word_counts', None)
                self.word_counts = wc if isinstance(wc, Counter) else Counter(wc or [])

                uc = _get(loaded_data, 'unigram_counts', None)
                self.unigram_counts = uc if isinstance(uc, Counter) else Counter(uc or [])

                # N-gram yapıları: dict -> defaultdict(Counter, ...)
                bc = _get(loaded_data, 'bigram_counts', None)
                if isinstance(bc, defaultdict) or isinstance(bc, dict):
                    self.bigram_counts = defaultdict(Counter, bc)
                else:
                    self.bigram_counts = defaultdict(Counter, dict(bc or {}))

                tc = _get(loaded_data, 'trigram_counts', None)
                if isinstance(tc, defaultdict) or isinstance(tc, dict):
                    self.trigram_counts = defaultdict(Counter, tc)
                else:
                    self.trigram_counts = defaultdict(Counter, dict(tc or {}))

                csb = _get(loaded_data, 'cross_sentence_bigrams', None)
                if isinstance(csb, defaultdict) or isinstance(csb, dict):
                    self.cross_sentence_bigrams = defaultdict(Counter, csb)
                else:
                    self.cross_sentence_bigrams = defaultdict(Counter, dict(csb or {}))

                cst = _get(loaded_data, 'cross_sentence_trigrams', None)
                if isinstance(cst, defaultdict) or isinstance(cst, dict):
                    self.cross_sentence_trigrams = defaultdict(Counter, cst)
                else:
                    self.cross_sentence_trigrams = defaultdict(Counter, dict(cst or {}))

                # Diğer alanlar
                tw = _get(loaded_data, 'total_words', 0)
                try:
                    self.total_words = int(tw)
                except Exception:
                    self.total_words = 0

                toks = _get(loaded_data, 'tokens', [])
                self.tokens = list(toks) if toks is not None else []

                sids = _get(loaded_data, 'sentence_ids', [])
                self.sentence_ids = list(sids) if sids is not None else []

                print(f"✅ Model yüklendi (eski pickle formatı veya uyumlu nesne: {type(loaded_data)})")
            except Exception as e:
                raise TypeError(f"❌ Eski format okunamadı: {e}")

        # YENİ FORMAT: Dict olarak kaydedilmiş
        elif isinstance(loaded_data, dict):
            print("🔄 Yeni format tespit edildi...")

            self.word_counts = loaded_data['word_counts']
            self.unigram_counts = loaded_data['unigram_counts']
            self.bigram_counts = defaultdict(Counter, loaded_data['bigram_counts'])
            self.trigram_counts = defaultdict(Counter, loaded_data['trigram_counts'])
            self.cross_sentence_bigrams = defaultdict(Counter, loaded_data['cross_sentence_bigrams'])
            self.cross_sentence_trigrams = defaultdict(Counter, loaded_data['cross_sentence_trigrams'])
            self.total_words = loaded_data['total_words']
            self.tokens = loaded_data['tokens']
            self.sentence_ids = loaded_data['sentence_ids']

            print(f"✅ Model yüklendi (yeni pickle formatı)")

        else:
            raise TypeError(f"❌ Bilinmeyen pickle formatı: {type(loaded_data)}")

        # İstatistikleri göster
        print(f"  - Benzersiz kelime: {len(self.word_counts)}")
        print(f"  - Toplam kelime: {self.total_words}")
        print(f"  - 2-gram sayısı: {len(self.bigram_counts)}")
        print(f"  - 3-gram sayısı: {len(self.trigram_counts)}")
    
    def _load_transformer(self):
        """Transformer modelini yükle (varsa)."""
        model_path = 'checkpoints/best_model.pth'
        tokenizer_path = 'tokenizer.json'
        
        if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
            print("ℹ️  Transformer model bulunamadı, sadece n-gram kullanılacak")
            return
        
        try:
            # Tokenizer yükle
            self.transformer_tokenizer = SimpleTokenizer()
            self.transformer_tokenizer.load(tokenizer_path)
            
            # Model yükle
            checkpoint = torch.load(model_path, map_location=self.device)
            config_dict = checkpoint['config']
            config = TransformerConfig(**config_dict)
            
            self.transformer_model = TurkishGPT(config).to(self.device)
            self.transformer_model.load_state_dict(checkpoint['model_state_dict'])
            self.transformer_model.eval()
            
            print("✓ Transformer model yüklendi")
        except Exception as e:
            print(f"⚠️  Transformer yükleme hatası: {e}")
            self.transformer_model = None
            self.transformer_tokenizer = None

    def get_probabilities(self, full_text):
        """
        Verilen metin için 1-gram, 2-gram ve 3-gram olasılıklarını hesaplar.
        Cümle bağlamı desteği eklenmiş.
        """
        if not full_text:  
            # Boş metin - sadece 1-gram göster
            unigram_probs = self._format_probabilities(self.unigram_counts, use_total=True)
            return {
                "unigram": unigram_probs,
                "bigram": [],
                "trigram": [],
                "current_word": "",
                "context": "",
                "is_new_sentence": False
            }

        words_list = full_text.strip().split()
        
        if len(words_list) == 0:
            unigram_probs = self._format_probabilities(self.unigram_counts, use_total=True)
            return {
                "unigram": unigram_probs,
                "bigram": [],
                "trigram": [],
                "current_word": "",
                "context": "",
                "is_new_sentence": False
            }
        
        # Son kelime
        last_word = words_list[-1].lower()
        current_word = strip_punctuation(last_word)
        
        # Cümle sonu kontrolü (SADECE .!? için)
        is_new_sentence = is_sentence_ending(last_word)
        
        result = {
            "unigram": [],
            "bigram": [],
            "trigram": [],
            "current_word": current_word,
            "context": "",
            "is_new_sentence": is_new_sentence
        }
        
        # 1-GRAM (UNIGRAM) olasılıkları
        result["unigram"] = self._format_probabilities(self.unigram_counts, use_total=True)
        
        if is_new_sentence:
            # Yeni cümle başlıyor - cross-sentence n-gram kullan
            # Cross-sentence bigram
            if current_word in self.cross_sentence_bigrams:
                result["bigram"] = self._format_probabilities(
                    self.cross_sentence_bigrams[current_word]
                )
            
            # Cross-sentence trigram
            if len(words_list) >= 2:
                prev_word = strip_punctuation(words_list[-2].lower())
                prev_context = (prev_word, current_word)
                
                if prev_context in self.cross_sentence_trigrams:
                    result["context"] = f"{prev_word} {current_word} [.]"
                    result["trigram"] = self._format_probabilities(
                        self.cross_sentence_trigrams[prev_context]
                    )
        else:
            # Normal n-gram (aynı cümle içinde)
            # 2-GRAM olasılıkları
            bigram_candidates = self.bigram_counts.get(current_word)
            if bigram_candidates:
                result["bigram"] = self._format_probabilities(bigram_candidates)
            
            # 3-GRAM olasılıkları
            if len(words_list) >= 2:
                prev_word = strip_punctuation(words_list[-2].lower())
                result["context"] = f"{prev_word} {current_word}"
                
                trigram_candidates = self.trigram_counts.get((prev_word, current_word))
                if trigram_candidates:
                    result["trigram"] = self._format_probabilities(trigram_candidates)
        
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
        return probabilities[:5]

    def predict(self, full_text, mode="suffix", use_transformer=False):
        """
        Kelime tamamlama veya cümle tamamlama.
        
        Args:
            full_text: Kullanıcının yazdığı metin
            mode: "suffix" (kelime tamamla) veya "completion" (cümle tamamla)
            use_transformer: True ise transformer kullan (varsa)
            
        Returns:
            Tahmin edilen metin
        """
        # Transformer kullan (varsa ve isteniyorsa)
        if use_transformer and self.transformer_model is not None and mode == "completion":
            return self._transformer_complete(full_text)
        
        # N-gram fallback
        if mode == "completion":
            return self._complete_sentence(full_text)
        else:
            return self._complete_word(full_text)
    
    def _transformer_complete(self, full_text, max_new_tokens=30):
        """Transformer ile cümle tamamlama."""
        if not full_text or not TORCH_AVAILABLE:
            return ""
        
        try:
            input_ids = self.transformer_tokenizer.encode(full_text, add_special_tokens=False)
            input_ids = torch.tensor([input_ids]).to(self.device)
            
            generated_ids = self.transformer_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_k=40,
                top_p=0.9,
            )
            
            all_text = self.transformer_tokenizer.decode(generated_ids[0].cpu().tolist())
            new_text = all_text[len(full_text):].strip()
            
            return new_text
        except Exception as e:
            print(f"⚠️  Transformer generation hatası: {e}")
            return self._complete_sentence(full_text)
    
    def _complete_word(self, full_text):
        """Kelime tamamlama."""
        if not full_text or full_text.endswith(" "):
            return ""

        words_list = full_text.split()
        current_partial = words_list[-1].lower()
        current_partial_clean = strip_punctuation(current_partial)
        
        matches = [w for w in self.word_counts if w.startswith(current_partial_clean)]
        if not matches:
            return ""

        best_completion = max(matches, key=lambda w: self.word_counts[w])
        suffix = best_completion[len(current_partial_clean):]
        
        if len(words_list) >= 2:
            prev_word = strip_punctuation(words_list[-2].lower())
            current_word = best_completion
            
            next_word_candidates = self.trigram_counts.get((prev_word, current_word))
            
            if next_word_candidates:
                words = list(next_word_candidates.keys())
                counts = list(next_word_candidates.values())
                chosen_next_word = random.choices(words, weights=counts, k=1)[0]
                return suffix + " " + chosen_next_word
        
        next_word_candidates = self.bigram_counts.get(best_completion)
        
        if next_word_candidates:
            words = list(next_word_candidates.keys())
            counts = list(next_word_candidates.values())
            chosen_next_word = random.choices(words, weights=counts, k=1)[0]
            return suffix + " " + chosen_next_word
        
        return suffix
    
    def _complete_sentence(self, full_text, max_words=6):
        """Cümle tamamlama."""
        if not full_text:
            return ""
        
        words_list = full_text.strip().split()
        if len(words_list) == 0:
            return ""
        
        words_clean = [strip_punctuation(w.lower()) for w in words_list]
        
        generated = []
        context = words_clean[-2:] if len(words_clean) >= 2 else words_clean[-1:]
        
        for _ in range(max_words):
            next_word = self._predict_next_word(context)
            
            if not next_word:
                break
            
            generated.append(next_word)
            context.append(next_word)
            context = context[-2:]
        
        return " ".join(generated)
    
    def _predict_next_word(self, context):
        """Sonraki kelimeyi tahmin et."""
        if len(context) >= 2:
            key = (context[-2].lower(), context[-1].lower())
            candidates = self.trigram_counts.get(key)
            
            if candidates:
                return self._weighted_random_choice(candidates)
        
        if len(context) >= 1:
            key = context[-1].lower()
            candidates = self.bigram_counts.get(key)
            
            if candidates:
                return self._weighted_random_choice(candidates)
        
        if self.unigram_counts:
            return self._weighted_random_choice(self.unigram_counts)
        
        return None
    
    def _weighted_random_choice(self, candidates):
        """Frekansa dayalı rastgele seçim."""
        if not candidates:
            return None
        
        words = list(candidates.keys())
        counts = list(candidates.values())
        
        return random.choices(words, weights=counts, k=1)[0]


# ===== YENİ: Model eğitme ve kaydetme scripti =====
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # python model.py train
        print("\n" + "="*60)
        print("🧠 MODEL EĞİTİMİ VE KAYDETME")
        print("="*60 + "\n")
        
        # story.txt'den eğit
        ai = NgramPredictor(filename="story.txt")
        
        # Pickle'a kaydet
        ai.save_to_pickle("saved_model.pkl")
        
        print("\n✅ Model eğitildi ve kaydedildi!")
        print("   Artık app.py hızlıca yükleyecek.\n")
    else:
        # python model.py (test)
        print("\n🧪 Model Test\n")
        
        # Pickle'dan yükle
        ai = NgramPredictor(load_from_pickle="saved_model.pkl")
        
        # Test et
        test_text = "bugün hava"
        result = ai.predict(test_text, mode="suffix")
        print(f"\nTest: '{test_text}' -> '{result}'")