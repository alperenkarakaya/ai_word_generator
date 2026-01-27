"""
Metin temizleme ve Noktalama Token Sistemi (TR001-TR018)
"""
import re
import unicodedata


# Noktalama token mapping
PUNCTUATION_TOKENS = {
    '.': 'TR001',   # Nokta (cümle sonu)
    ',': 'TR002',   # Virgül
    '!': 'TR003',   # Ünlem (cümle sonu)
    '?': 'TR004',   # Soru işareti (cümle sonu)
    ';': 'TR005',   # Noktalı virgül
    ':': 'TR006',   # İki nokta
    '-': 'TR007',   # Tire
    '(': 'TR010',   # Açma parantez
    ')': 'TR011',   # Kapama parantez
    '"': 'TR012',   # Tırnak
    "'": 'TR013',   # Tek tırnak
    '...': 'TR016', # Üç nokta
    '\n': 'TR017',  # Satır sonu
}

# Ters mapping (TR001 -> .)
TOKEN_TO_PUNCTUATION = {v: k for k, v in PUNCTUATION_TOKENS.items()}

# Cümle bitirici tokenlar (SADECE bunlar)
SENTENCE_ENDING_TOKENS = {'TR001', 'TR003', 'TR004'}  # . ! ?


def normalize_turkish_text(text: str) -> str:
    """Türkçe metni normalize eder (NFC normalizasyonu)."""
    return unicodedata.normalize('NFC', text)


def replace_punctuation_with_tokens(text: str) -> str:
    """
    Noktalama işaretlerini token'lara çevirir.
    Tokenlar kelimeye bitişik olur ama model için ayrı kelime olarak algılanır.
    Örnek: "Merhaba, dünya!" -> "MerhabaTR002 dünyaTR003"
    """
    # Önce üç noktayı değiştir (öncesinde boşluk yok, sonrasında var)
    text = text.replace('...', 'TR016 ')
    
    # Diğer noktalamalar (öncesinde boşluk yok, sonrasında var)
    for punct, token in sorted(PUNCTUATION_TOKENS.items(), key=lambda x: -len(x[0])):
        if punct == '...':
            continue
        # Noktalama işaretini kelimeye bitişik token ile değiştir
        text = text.replace(punct, f'{token} ')
    
    return text


def restore_punctuation_from_tokens(text: str) -> str:
    """
    Token'ları noktalama işaretlerine geri çevirir.
    Örnek: "MerhabaTR002 dünyaTR003" -> "Merhaba, dünya! "
    """
    for token, punct in TOKEN_TO_PUNCTUATION.items():
        # Önce boşluklu halini dene (eski format uyumluluğu için)
        text = text.replace(f' {token} ', punct + ' ')
        # Bitişik token halini değiştir (kelimeye bitişik token + boşluk -> noktalama + boşluk)
        text = text.replace(f'{token} ', punct + ' ')
        # Sonda kalan token varsa (boşluksuz)
        if text.endswith(token):
            text = text[:-len(token)] + punct
    return text


def clean_special_characters(text: str) -> str:
    """
    Özel karakterleri temizler.
    SADECE Türkçe harfler, rakamlar, boşluklar ve TRxxx tokenları kalır.
    """
    # E-posta ve URL'leri kaldır
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    
    # TRxxx tokenlarını geçici koru
    tr_tokens = re.findall(r'TR\d{3}', text)
    placeholders = {}
    for i, token in enumerate(tr_tokens):
        placeholder = f'___PH{i:04d}___'
        placeholders[placeholder] = token
        text = text.replace(token, placeholder, 1)
    
    # Sadece Türkçe harfler, rakamlar, boşluk ve placeholder
    allowed = set('abcçdefgğhıijklmnoöprsştuüvyzwxqABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZWXQ0123456789 _')
    text = ''.join(c for c in text if c in allowed)
    
    # Placeholder'ları geri koy
    for ph, token in placeholders.items():
        text = text.replace(ph, token)
    
    return text


def full_clean(text: str, lowercase: bool = True, use_tokens: bool = True) -> str:
    """
    Metni tam olarak temizler.
    
    Args:
        text: Ham metin
        lowercase: Küçük harfe çevir
        use_tokens: Noktalama tokenları kullan (TR001, TR002 vb.)
    """
    # 1. Normalize
    text = normalize_turkish_text(text)
    
    # 2. Wikipedia/özel format karakterlerini temizle
    text = text.replace('@-@', '-')  # @ - @ → -
    text = text.replace('@ - @', '-')
    text = text.replace('@.@', '.')  # @ . @ → .
    text = text.replace('@ . @', '.')
    text = text.replace('@,@', ',')  # @ , @ → ,
    text = text.replace('@ , @', ',')
    text = re.sub(r'@\s*-\s*@', '-', text)  # Tüm varyasyonları
    text = re.sub(r'@\s*\.\s*@', '.', text)
    text = re.sub(r'@\s*,\s*@', ',', text)
    # Kalan @ işaretlerini kaldır
    text = text.replace('@', '')
    
    # 3. Küçük harf
    if lowercase:
        text = text.lower()
    
    # 4. Noktalama → Token
    if use_tokens:
        text = replace_punctuation_with_tokens(text)
    
    # 5. Özel karakterleri temizle
    text = clean_special_characters(text)
    
    # 6. Çoklu boşluk ve gereksiz tire/çizgileri temizle
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'-\s+-', '-', text)  # - - → -
    
    return text


def is_sentence_ending(token: str) -> bool:
    """Token cümle bitirici mi? (TR001, TR003, TR004)"""
    return token in SENTENCE_ENDING_TOKENS


def is_punctuation_token(token: str) -> bool:
    """Token noktalama tokeni mi? (TR001-TR018)"""
    return token.startswith('TR') and len(token) == 5 and token[2:].isdigit()


def split_sentences(text: str) -> list:
    """Metni cümlelere böler (SADECE TR001, TR003, TR004'ten)"""
    pattern = r'\s*(?:TR001|TR003|TR004)\s*'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize_with_sentences(text: str) -> tuple:
    """
    Metni tokenize eder ve her tokenin cümle ID'sini döndürür.
    
    Returns:
        (tokens, sentence_ids)
    """
    tokens = []
    sentence_ids = []
    current_sentence_id = 0
    
    words = text.split()
    
    for word in words:
        tokens.append(word)
        sentence_ids.append(current_sentence_id)
        
        if word in SENTENCE_ENDING_TOKENS:
            current_sentence_id += 1
    
    return tokens, sentence_ids


# Test
if __name__ == "__main__":
    test = "Bugün hava çok güzel! Ali okula gitti."
    
    print("Orijinal:", test)
    cleaned = full_clean(test, use_tokens=True)
    print("Tokenlu:", cleaned)
    
    tokens, sent_ids = tokenize_with_sentences(cleaned)
    print("\nTokenlar:")
    for token, sid in zip(tokens, sent_ids):
        ending = "🔴" if is_sentence_ending(token) else "  "
        print(f"  {ending} {token:15} -> cümle {sid}")
    
    restored = restore_punctuation_from_tokens(cleaned)
    print("\nRestore:", restored)