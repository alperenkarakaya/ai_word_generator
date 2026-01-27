import pickle
from model_new import NgramPredictor

# --- AYAR: Bilgisayarı korumak için veri kısıtlaması ---
# Dosyanın tamamını okumak yerine sadece başından 50.000 karakter alacağız.
# Böylece işlem saniyeler sürecek ve PC kilitlenmeyecek.

print("Veri hazırlanıyor (kırpılıyor)...")

original_file = "story_tokenized.txt"
temp_file = "training_sample.txt"

# 1. Büyük dosyadan sadece küçük bir parça oku (50 bin karakter)
try:
    with open(original_file, "r", encoding="utf-8") as f:
        partial_data = f.read(10000000) # Sayıyı artırırsan RAM kullanımı artar!
    
    # 2. Bu parçayı geçici bir dosyaya kaydet
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(partial_data)
        
    print(f"Büyük dosya yerine '{temp_file}' oluşturuldu ve kullanılacak.")

except FileNotFoundError:
    print(f"HATA: '{original_file}' bulunamadı! Dosya adının doğru olduğundan emin ol.")
    exit()

# -------------------------------------------------------

print("Model eğitiliyor, lütfen bekleyin...")

# ARTIK DEVASA DOSYAYI DEĞİL, KÜÇÜK DOSYAYI VERİYORUZ:
ai = NgramPredictor(temp_file) 

print("Model kaydediliyor...")
with open("saved_model.pkl", "wb") as f:
    pickle.dump(ai, f)

print("Bitti! Artık app.py çalıştırabilirsin.")