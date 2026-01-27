"""
Story.txt'deki noktalama işaretlerini ve sayıları özel tokenlerle değiştirme modülü
"""
import re

# Noktalama işaretleri ve özel karakterler için token tanımları
PUNCTUATION_TOKENS = {
    '.': 'TR001',   # Nokta (cümle sonu)
    ',': 'TR002',   # Virgül
    '!': 'TR003',   # Ünlem
    '?': 'TR004',   # Soru işareti
    ';': 'TR005',   # Noktalı virgül
    ':': 'TR006',   # İki nokta
    '-': 'TR007',   # Tire
    '—': 'TR008',   # Uzun tire
    '–': 'TR009',   # Orta tire
    '(': 'TR010',   # Açma parantez
    ')': 'TR011',   # Kapama parantez
    '"': 'TR012',   # Tırnak
    "'": 'TR013',   # Tek tırnak
    '«': 'TR014',   # Açma tırnak
    '»': 'TR015',   # Kapama tırnak
    '...': 'TR016', # Üç nokta
    '\n': 'TR017',  # Satır sonu
    '\t': 'TR018',  # Tab
}

# Ters mapping (token'dan karaktere)
TOKEN_TO_PUNCT = {v: k for k, v in PUNCTUATION_TOKENS.items()}

# Sayılar için token
NUMBER_TOKEN = 'NUM'


def text_to_tokens(text):
    """
    Metindeki noktalama işaretlerini ve sayıları token'lara çevirir.
    """
    # Önce üç noktayı değiştir (tek nokta ile karışmasın)
    text = text.replace('...', ' TR016 ')
    
    # Diğer noktalama işaretlerini değiştir
    for punct, token in PUNCTUATION_TOKENS.items():
        if punct not in ['...', '\n', '\t']:  # Bunları zaten hallettik veya özel işlem gerekiyor
            text = text.replace(punct, f' {token} ')
    
    # Satır sonlarını ve tab'leri işle
    text = text.replace('\n', ' TR017 ')
    text = text.replace('\t', ' TR018 ')
    
    # Sayıları NUM token'ı ile değiştir (opsiyonel olarak sayıları tutabiliriz)
    # Şimdilik sayıları koruyalım ama gelecekte NUM123 gibi özel tokenler eklenebilir
    # text = re.sub(r'\b\d+\b', NUMBER_TOKEN, text)
    
    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def tokens_to_text(tokenized_text):
    """
    Token'ları orijinal metne geri çevirir.
    """
    text = tokenized_text
    
    # Token'ları noktalama işaretlerine çevir
    for token, punct in TOKEN_TO_PUNCT.items():
        if token == 'TR017':  # Satır sonu
            text = text.replace(f' {token} ', punct)
        elif token == 'TR018':  # Tab
            text = text.replace(f' {token} ', punct)
        else:
            text = text.replace(f' {token} ', punct)
            text = text.replace(f'{token} ', punct)
            text = text.replace(f' {token}', punct)
    
    # Bazı noktalama işaretlerinden önce/sonra boşluk düzelt
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Önündeki boşluğu sil
    text = re.sub(r'([.,!?;:])', r'\1 ', text)     # Sonuna boşluk ekle
    text = re.sub(r'\s+', ' ', text)                # Fazla boşlukları temizle
    text = text.strip()
    
    return text


def process_story_file(input_file, output_file):
    """
    Story dosyasını tokenize eder ve yeni dosyaya yazar.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            original_text = f.read()
        
        print(f"✓ {input_file} okundu ({len(original_text)} karakter)")
        
        # Tokenize et
        tokenized = text_to_tokens(original_text)
        
        # Kaydet
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(tokenized)
        
        print(f"✓ {output_file} oluşturuldu ({len(tokenized)} karakter)")
        print(f"\nİlk 500 karakter önizleme:")
        print(tokenized[:500])
        
        return True
    
    except FileNotFoundError:
        print(f"HATA: {input_file} bulunamadı!")
        return False
    except Exception as e:
        print(f"HATA: {e}")
        return False


if __name__ == '__main__':
    # story.txt'yi tokenize edip story_tokenized.txt'ye kaydet
    process_story_file('story.txt', 'story_tokenized.txt')
