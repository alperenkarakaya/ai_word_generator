from datasets import load_dataset
import os

# Veri setini indiriyoruz
print("Veri seti indiriliyor...")
dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")

# Dosya ismini senin istediğin gibi "story.txt" yapıyoruz
file_name = "story.txt"

# Dosyanın tam olarak scriptin çalıştığı klasöre kaydedilmesi için yol birleştiriyoruz
current_directory = os.getcwd()
full_path = os.path.join(current_directory, file_name)

print(f"Metinler şu dosyaya yazılıyor: {file_name}")

with open(file_name, "w", encoding="utf-8") as f:
    for item in dataset:
        # Boş satırları atlayarak sadece dolu metinleri yazalım
        text = item['text'].strip()
        if text:
            f.write(text + "\n")

print(f"İşlem bitti! Toplam {len(dataset)} satır işlendi.")
print(f"Dosyanın konumu: {full_path}")