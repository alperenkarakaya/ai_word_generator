import pickle
import re
import os
import text_utils
from model import NgramPredictor  # DÜZELTİLDİ: model_new yerine model

# --- AYARLAR ---
INPUT_FILE = "story.txt"
TEMP_TOKENIZED_FILE = "story_tokenized_temp.txt"
MODEL_OUTPUT_FILE = "saved_model.pkl"

# Performans ayarı: 1 Milyon karakter
READ_LIMIT = 1000000 

def preprocess_and_tokenize(text):
    """Noktalama işaretlerini TR kodlarına çevirir."""
    print("Metin tokenlarına ayrılıyor (Regex)...")
    
    # Önce Wikipedia/özel format karakterlerini temizle
    text = text.replace('@-@', '-')
    text = text.replace('@ - @', '-')
    text = text.replace('@.@', '.')
    text = text.replace('@ . @', '.')
    text = text.replace('@,@', ',')
    text = text.replace('@ , @', ',')
    text = re.sub(r'@\s*-\s*@', '-', text)
    text = re.sub(r'@\s*\.\s*@', '.', text)
    text = re.sub(r'@\s*,\s*@', ',', text)
    # Kalan @ işaretlerini kaldır
    text = text.replace('@', '')
    
    mapping = text_utils.PUNCTUATION_TOKENS
    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
    pattern = '|'.join(map(re.escape, sorted_keys))
    
    def replace_func(match):
        token = mapping[match.group(0)]
        return f" {token} "

    processed_text = re.sub(pattern, replace_func, text)
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    return processed_text

# --- ANA İŞLEM ---

print(f"1. '{INPUT_FILE}' okunuyor...")

try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        if READ_LIMIT > 0:
            raw_data = f.read(READ_LIMIT)
            print(f"   Güvenlik modu: Sadece ilk {READ_LIMIT} karakter okundu.")
        else:
            raw_data = f.read()
            print("   Dosyanın tamamı okundu.")

    # Tokenizer işlemini uygula
    tokenized_data = preprocess_and_tokenize(raw_data)

    # İşlenmiş veriyi kaydet
    print(f"2. Tokenlanmış veri '{TEMP_TOKENIZED_FILE}' dosyasına yazılıyor...")
    with open(TEMP_TOKENIZED_FILE, "w", encoding="utf-8") as f:
        f.write(tokenized_data)

    # Modeli eğit
    print("3. Ngram Modeli eğitiliyor...")
    
    # DÜZELTME: Modeli boş başlatıp, dosyadan eğit diyoruz
    ai = NgramPredictor(load_from_pickle=None)
    ai.train_from_file(TEMP_TOKENIZED_FILE)

    # Modeli kaydet
    print(f"4. Model '{MODEL_OUTPUT_FILE}' olarak kaydediliyor...")
    
    # Tüm sınıfı değil, sadece verileri kaydedelim (daha güvenli)
    model_data = {
        'word_counts': ai.word_counts,
        'unigram_counts': ai.unigram_counts,
        'bigram_counts': dict(ai.bigram_counts), # defaultdict pickle hatası verebilir, dict'e çevirdik
        'trigram_counts': dict(ai.trigram_counts),
        'total_words': ai.total_words
    }
    
    with open(MODEL_OUTPUT_FILE, "wb") as f:
        pickle.dump(model_data, f)

    print("\n✅ İŞLEM BAŞARILI!")
    print(f"Artık 'python app.py' çalıştırabilirsin.")

except FileNotFoundError:
    print(f"❌ HATA: '{INPUT_FILE}' bulunamadı! Dosya adını kontrol et.")
except Exception as e:
    print(f"❌ BEKLENMEYEN HATA: {e}")