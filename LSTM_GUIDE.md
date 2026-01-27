# 🧠 AI Word Generator - LSTM Model Kılavuzu

## 🎯 Nedir?

Bu proje artık **iki farklı model** destekliyor:

1. **N-gram Model** (mevcut) - Hızlı, basit, küçük boyutlu
2. **LSTM Model** (yeni) - Daha akıllı, daha doğal metin üretimi

## 📦 Gereksinimler

### N-gram için (mevcut)
```bash
pip install flask
```

### LSTM için (yeni)
```bash
pip install torch flask
```

**Not:** PyTorch yükleme Windows için:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 🚀 Kullanım

### 1. N-gram Model (Mevcut - Zaten Çalışıyor)

```bash
# Model oluştur
python create_pickle.py

# Uygulamayı başlat
python app.py
```

### 2. LSTM Model (Yeni - Daha Güçlü)

```bash
# LSTM modelini eğit (5-10 dakika sürer)
python train_lstm.py

# Test et
python train_lstm.py test

# Flask ile kullan (LSTM otomatik yüklenir)
python app.py
```

## 🔥 LSTM vs N-gram Karşılaştırması

| Özellik | N-gram | LSTM |
|---------|--------|------|
| **Hız** | ⚡ Çok hızlı | 🐌 Orta |
| **Kalite** | 📊 İyi | 🎯 Çok iyi |
| **Model Boyutu** | 💾 ~5 MB | 💾 ~50 MB |
| **Eğitim Süresi** | ⏱️ 5 saniye | ⏱️ 5-10 dakika |
| **Bellek** | 🧠 ~50 MB | 🧠 ~500 MB |
| **Doğal Dil** | 📝 Orta | 📝 Çok iyi |

## 💡 Hangi Model Ne Zaman?

### N-gram Kullan:
- ✅ Hızlı prototip
- ✅ Düşük kaynak
- ✅ Basit kelime tamamlama
- ✅ Küçük veri setleri

### LSTM Kullan:
- ✅ Daha akıllı tahminler
- ✅ Uzun bağlam (context)
- ✅ Doğal cümle üretimi
- ✅ Büyük veri setleri

## 🎮 Frontend Kullanımı

### Tab Tuşu (N-gram)
- Kelime tamamlama
- Hızlı, anında

### Shift+Tab (N-gram veya LSTM)
- Cümle tamamlama
- LSTM varsa otomatik kullanır

Frontend'de LSTM kullanımı için:
```javascript
// main.js içinde zaten hazır
// use_lstm=true parametresi ile çağrılıyor
```

## 📊 Eğitim Hyperparameters (İleri Seviye)

`train_lstm.py` içinde değiştirebilirsiniz:

```python
predictor.train(
    seq_length=100,      # Daha uzun bağlam için artır (50-200)
    embedding_dim=128,   # Daha zengin temsil (64-256)
    hidden_dim=256,      # Daha fazla kapasite (128-512)
    num_layers=2,        # Daha derin model (1-4)
    epochs=5,            # Daha iyi eğitim (5-20)
    batch_size=64,       # GPU varsa artır (32-128)
    learning_rate=0.001  # Daha dikkatli öğrenme (0.0001-0.01)
)
```

## 🔧 Hybrid Mod

Uygulama artık **hybrid mod** destekliyor:

1. `lstm_model.pth` varsa → LSTM kullanır
2. Yoksa → N-gram fallback

Kod içinde:
```python
# app.py otomatik tespit ediyor
ai = NgramPredictor(
    load_from_pickle="saved_model.pkl",
    lstm_model_path="lstm_model.pth"  # opsiyonel
)
```

## 📈 Performans İpuçları

### LSTM Hızlandırma:
1. **GPU kullan** (CUDA kurulu ise)
2. **Batch size artır** (GPU belleğine göre)
3. **Sequence length azalt** (50-100 yeterli)

### N-gram Hızlandırma:
1. **Pickle dosyasını cache'le**
2. **Trigram öncelik ver**
3. **Token sistemi kullan**

## 🐛 Sorun Giderme

### "PyTorch yok" hatası
```bash
pip install torch
```

### "CUDA out of memory" hatası
```python
# train_lstm.py içinde:
batch_size=32  # veya 16
```

### "LSTM yüklenemedi" uyarısı
- Normal, N-gram kullanılacak
- LSTM istiyorsanız: `python train_lstm.py`

## 📚 Daha Fazla Bilgi

- LSTM nedir? → [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- PyTorch tutorialları → [PyTorch Tutorials](https://pytorch.org/tutorials/)
- Character-level LM → [Andrej Karpathy Blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

## 🎯 Sonraki Adımlar

1. ✅ N-gram modeli çalışıyor
2. 🔄 LSTM modelini eğitin
3. 🧪 İkisini test edin
4. 🚀 Hybrid mod ile en iyisini kullanın

---

**Not:** LSTM eğitimi için GPU önerilir ama gerekli değildir. CPU ile de çalışır, sadece daha yavaş olur (5-10 dakika vs 2-3 dakika).
