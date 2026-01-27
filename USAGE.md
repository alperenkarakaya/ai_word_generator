# Sentence-Based N-gram Model - Kullanım Kılavuzu

## Yapılan Değişiklikler

### 1. Token Sistemi
Noktalama işaretleri ve özel karakterler artık özel tokenlerle temsil ediliyor:

- `.` → `TR001` (Nokta - cümle sonu)
- `,` → `TR002` (Virgül)
- `!` → `TR003` (Ünlem)
- `?` → `TR004` (Soru işareti)
- `;` → `TR005` (Noktalı virgül)
- `:` → `TR006` (İki nokta)
- `-` → `TR007` (Tire)
- `—` → `TR008` (Uzun tire)
- `–` → `TR009` (Orta tire)
- `(` → `TR010` (Açma parantez)
- `)` → `TR011` (Kapama parantez)
- `"` → `TR012` (Tırnak)
- `'` → `TR013` (Tek tırnak)
- `«` → `TR014` (Açma tırnak)
- `»` → `TR015` (Kapama tırnak)
- `...` → `TR016` (Üç nokta)
- `\n` → `TR017` (Satır sonu)
- `\t` → `TR018` (Tab)

### 2. Sentence-Based Yaklaşım
- Artık kelime kelime değil, cümle bazlı çalışıyor
- Noktalama işaretleri korunuyor ve anlamlı şekilde kullanılıyor
- Sayılar ve diğer özel karakterler kaybolmuyor
- Her cümle bitirici noktalama işaretinden (. ! ?) sonra model yeni cümleye devam edebiliyor

## Kurulum ve Kullanım

### Adım 1: story.txt'yi Tokenize Etme
```bash
python preprocess_story.py
```

Bu komut:
- `story.txt` dosyasını okur
- Tüm noktalama işaretlerini özel tokenlerle (TR001, TR002, vb.) değiştirir
- `story_tokenized.txt` dosyasını oluşturur

### Adım 2: Uygulamayı Çalıştırma
```bash
python app.py
```

Uygulama `story_tokenized.txt` dosyasını kullanarak model oluşturur.

## Dosya Yapısı

```
project/
│
├── story.txt                  # Orijinal story dosyası
├── story_tokenized.txt        # Tokenize edilmiş story dosyası
│
├── preprocess_story.py        # Tokenizasyon modülü
├── model_new.py              # Yeni sentence-based model
├── model.py                  # Eski word-based model (yedek)
├── app.py                    # Flask uygulaması
│
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
│
└── USAGE.md                  # Bu dosya
```

## Nasıl Çalışır?

### Tokenizasyon
1. `preprocess_story.py` story.txt'deki tüm noktalama işaretlerini tarar
2. Her noktalama işaretini karşılık gelen token ile değiştirir
3. Sonuç: "Merhaba dünya." → "merhaba dünya TR001"

### Model Eğitimi
1. `model_new.py` tokenize edilmiş metni okur
2. N-gram modelleri oluşturur (token'ları da içerecek şekilde)
3. Örnek: "dünya TR001" kelime çiftinden sonra ne geleceğini öğrenir

### Tahmin ve Detokenizasyon
1. Kullanıcı input girer: "merhaba"
2. Model tahmin yapar: "dünya TR001"
3. Detokenizasyon: "dünya TR001" → "dünya."
4. Kullanıcıya çıktı: "dünya."

## Örnek Kullanım

### Python'da Token Dönüşümü
```python
from preprocess_story import text_to_tokens, tokens_to_text

# Metni tokenize et
text = "Merhaba! Nasılsın?"
tokenized = text_to_tokens(text)
print(tokenized)
# Çıktı: "merhaba TR003 nasılsın TR004"

# Token'ları geri çevir
original = tokens_to_text(tokenized)
print(original)
# Çıktı: "merhaba! nasılsın?"
```

### Model ile Tahmin
```python
from model_new import NgramPredictor

model = NgramPredictor("story_tokenized.txt")

# Cümle devam ettirme
prediction = model.predict("merhaba ")
print(prediction)  # Noktalama işaretleri dahil gelir

# Olasılık hesaplama
probs = model.get_probabilities("merhaba dünya")
print(probs)  # Detokenize edilmiş sonuçlar
```

## Avantajlar

✅ **Noktalama Korunur**: Artık noktalama işaretleri kaybolmuyor
✅ **Sayılar Korunur**: Sayılar doğru şekilde işleniyor
✅ **Cümle Yapısı**: Model cümle yapısını öğreniyor
✅ **Daha Doğal**: Üretilen metin daha doğal ve okunabilir
✅ **Geriye Dönük**: Token'lar kolayca orijinal metne çevrilebilir

## Test Etme

Story dosyasını tokenize ettikten sonra:

1. Web arayüzünü aç: http://localhost:5000
2. Bir kelime yazmaya başla
3. Model artık noktalama işaretleri ve sayılarla birlikte tahmin yapacak
4. Cümle bitirici noktalama (. ! ?) geldiğinde yeni cümle başlatacak

## Sorun Giderme

**Problem**: `story_tokenized.txt bulunamadı` hatası
**Çözüm**: Önce `python preprocess_story.py` komutunu çalıştırın

**Problem**: Tokenler çıktıda görünüyor (TR001 gibi)
**Çözüm**: `model_new.py` detokenizasyon yapıyor, `model.py` yerine `model_new.py` kullanıldığından emin olun

**Problem**: Noktalama hala kaybolıyor
**Çözüm**: `app.py` dosyasının `story_tokenized.txt` dosyasını kullandığından emin olun
