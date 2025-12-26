import random
from collections import defaultdict
import re

# --- AYARLAR ---
DOSYA_ADI = "story.txt"
# BURASI KRİTİK: Bu sayıyı 20-30 gibi yüksek yapmak, 
# modelin kelimeyi "başından sonuna kadar" hatırlamasını sağlar.
ADIM_SAYISI = 20 

def metni_hazirla(filename, n):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            data = file.read()
        
        data = data.lower()
        # Sadece harfleri al
        data = re.sub(r'[^a-z\s]', '', data)
        raw_words = data.split()
        
        # Her kelimenin başına ve sonuna 20 tane nokta koyuyoruz.
        # Bu sayede model "....................c" gördüğünde "kelime yeni başlıyor ve c ile başlıyor" der.
        padding = "." * n
        processed_words = [f"{padding}{w}{padding}" for w in raw_words]
        
        return processed_words
    except FileNotFoundError:
        print("Dosya bulunamadı.")
        return []

word_list = metni_hazirla(DOSYA_ADI, ADIM_SAYISI)

if word_list:
    # --- EĞİTİM ---
    markov_model = defaultdict(list)

    for word in word_list:
        # 20 karakterlik bloklar halinde öğreniyor
        for i in range(len(word) - ADIM_SAYISI):
            current_state = word[i : i + ADIM_SAYISI]
            next_char = word[i + ADIM_SAYISI]
            markov_model[current_state].append(next_char)

    # --- ÜRETİM ---
    def kelime_uret():
        # Başlangıç her zaman 20 tane nokta
        current_state = "." * ADIM_SAYISI
        result_word = ""
        
        while True:
            possibilities = markov_model.get(current_state)
            
            if not possibilities:
                break
            
            # Olasılıklar içinden seç (Eğer 'memories' ve 'memory' varsa, 'memor' köküne kadar ortak gelir, sonra ayrılır)
            next_char = random.choice(possibilities)
            
            if next_char == ".":
                break
            
            result_word += next_char
            
            # Pencereyi kaydır
            current_state = current_state[1:] + next_char
        
        return result_word

    # --- SONUÇLAR ---
    print(f"--- Güvenli Mod (Hafıza: {ADIM_SAYISI}) Aktif ---")
    print("Risk minimize edildi. Sadece hikaye yapısına uygun kelimeler üretiliyor:\n")
    
    # 10 tane örnek üretelim
    for i in range(10):
        print(f"> {kelime_uret().capitalize()}")

else:
    print("Veri yok.")