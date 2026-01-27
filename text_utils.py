"""
Metin temizleme ve normalizasyon araçları
"""
import re
import unicodedata
from typing import List, Tuple


def normalize_turkish_text(text: str) -> str:
    """
    Türkçe metni normalize eder (NFC normalizasyonu).
    
    Args:
        text: Ham metin
        
    Returns:
        Normalize edilmiş metin
    """
    # Unicode NFC normalizasyonu (Türkçe karakterler için önemli)
    text = unicodedata.normalize('NFC', text)
    return text


def clean_special_characters(text: str, keep_sentence_endings: bool = True) -> str:
    """
    Özel karakterleri temizler.
    SADECE cümle bitirici noktalama işaretlerini korur: . ! ?
    E-posta/URL içindeki noktaları korur.
    
    Args:
        text: Temizlenecek metin
        keep_sentence_endings: True ise .!? karakterlerini korur
        
    Returns:
        Temizlenmiş metin
    """
    # E-posta ve URL'leri geçici olarak koru (nokta problemini önler)
    email_url_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b|(?:http[s]?://)?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    emails_urls = re.findall(email_url_pattern, text)
    
    # E-posta/URL'leri placeholder ile değiştir
    for i, match in enumerate(emails_urls):
        text = text.replace(match, f' __EMAILURL{i}__ ')
    
    if keep_sentence_endings:
        # Cümle bitiricileri geçici olarak koru
        # Sadece boşluktan veya satır sonundan önce gelen noktaları koru
        text = re.sub(r'\.\s+', ' __PERIOD__ ', text)
        text = re.sub(r'\.($)', ' __PERIOD__ ', text)
        text = text.replace('!', ' __EXCLAMATION__ ')
        text = text.replace('?', ' __QUESTION__ ')
    
    # Tüm özel karakterleri kaldır (sadece harf, rakam, boşluk kalsın)
    text = re.sub(r'[^\w\s]', '', text)
    
    if keep_sentence_endings:
        # Cümle bitiricileri geri getir (kelimeye YAPIŞIK şekilde)
        text = text.replace(' __PERIOD__ ', '.')
        text = text.replace(' __EXCLAMATION__ ', '!')
        text = text.replace(' __QUESTION__ ', '?')
        
        text = text.replace('__PERIOD__', '.')
        text = text.replace('__EXCLAMATION__', '!')
        text = text.replace('__QUESTION__', '?')
    
    # E-posta/URL'leri geri getir (ama noktaları çıkar - zaten temizlendi)
    for i, match in enumerate(emails_urls):
        clean_match = re.sub(r'[^\w]', '', match)
        text = text.replace(f'__EMAILURL{i}__', clean_match)
    
    return text


def clean_whitespace(text: str) -> str:
    """
    Çoklu boşlukları tek boşluğa indirir ve baş/sondaki boşlukları temizler.
    Noktalama işaretlerinden sonra boşluk olmasını garantiler.
    
    Args:
        text: Temizlenecek metin
        
    Returns:
        Temizlenmiş metin
    """
    # Noktalama işaretlerinden sonra boşluk ekle (yoksa)
    text = re.sub(r'([.!?])([A-Za-zğüşıöçĞÜŞİÖÇ])', r'\1 \2', text)
    
    # Çoklu boşlukları tek boşluğa indir
    text = re.sub(r'\s+', ' ', text)
    
    # Noktalama işaretinden önceki boşlukları kaldır
    # "kelime ." -> "kelime."
    text = re.sub(r'\s+([.!?])', r'\1', text)
    
    # Baş ve sondaki boşlukları temizle
    text = text.strip()
    
    return text


def full_clean(text: str, lowercase: bool = True, keep_punctuation: bool = True) -> str:
    """
    Metni tam olarak temizler (tüm adımlar).
    
    Args:
        text: Ham metin
        lowercase: True ise küçük harfe çevir
        keep_punctuation: True ise cümle sonu noktalamalarını koru (.!?)
        
    Returns:
        Temizlenmiş metin
    """
    # 1. Normalize et
    text = normalize_turkish_text(text)
    
    # 2. Küçük harfe çevir (opsiyonel)
    if lowercase:
        text = text.lower()
    
    # 3. Özel karakterleri temizle
    text = clean_special_characters(text, keep_sentence_endings=keep_punctuation)
    
    # 4. Boşlukları temizle ve noktalamayı düzelt
    text = clean_whitespace(text)
    
    return text


def split_sentences(text: str) -> List[str]:
    """
    Metni cümlelere böler (SADECE .!? karakterlerinden).
    
    Args:
        text: Bölünecek metin
        
    Returns:
        Cümle listesi
    """
    # Sadece cümle bitirici noktalamalardan böl
    sentences = re.split(r'[.!?]+', text)
    # Boş cümleleri filtrele
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def tokenize_with_sentences(text: str) -> Tuple[List[str], List[int]]:
    """
    Metni tokenize eder ve her tokenin hangi cümleye ait olduğunu işaretler.
    Noktalama işaretleri kelimeye YAPIŞIK olarak işlenir.
    
    Args:
        text: Tokenize edilecek metin
        
    Returns:
        (tokens, sentence_ids) tuple
        - tokens: kelime listesi (noktalama yapışık: "gitti." gibi)
        - sentence_ids: her kelimenin hangi cümleye ait olduğu (0, 1, 2, ...)
    """
    tokens = []
    sentence_ids = []
    current_sentence_id = 0
    
    # Kelimeleri ayır (noktalama yapışık kalacak)
    words = text.split()
    
    for word in words:
        tokens.append(word)
        sentence_ids.append(current_sentence_id)
        
        # Eğer kelime cümle bitirici ile bitiyorsa, sonraki kelime yeni cümle
        if word.endswith('.') or word.endswith('!') or word.endswith('?'):
            current_sentence_id += 1
    
    return tokens, sentence_ids


def is_sentence_ending(token: str) -> bool:
    """
    Token cümle bitirici mi kontrol eder.
    
    Args:
        token: Kontrol edilecek kelime
        
    Returns:
        True ise cümle bitirici
    """
    return token.endswith('.') or token.endswith('!') or token.endswith('?')


def strip_punctuation(token: str) -> str:
    """
    Tokendan noktalama işaretini çıkarır.
    
    Args:
        token: "gitti." gibi kelime
        
    Returns:
        "gitti" gibi temiz kelime
    """
    return token.rstrip('.!?')


def get_sentence_context(tokens: List[str], sentence_ids: List[int], 
                         current_position: int, context_window: int = 3) -> List[str]:
    """
    Verilen pozisyon için cümle bağlamını döndürür.
    Eğer yeni bir cümle başlıyorsa, önceki cümlenin son N kelimesini döndürür.
    
    Args:
        tokens: Tüm tokenlar
        sentence_ids: Her tokenin cümle ID'si
        current_position: Mevcut pozisyon (tahmin yapılacak yer)
        context_window: Kaç kelime geriye bakılacak
        
    Returns:
        Bağlam kelimeleri listesi (noktalama temizlenmiş)
    """
    if current_position == 0:
        return []
    
    # Mevcut pozisyonun cümle ID'si
    if current_position < len(sentence_ids):
        current_sentence_id = sentence_ids[current_position]
    else:
        # Metin sonundaysak, son cümlenin ID'sini al
        current_sentence_id = sentence_ids[-1] if sentence_ids else 0
    
    # Önceki pozisyonun cümle ID'si
    prev_sentence_id = sentence_ids[current_position - 1] if current_position > 0 else -1
    
    # Yeni cümle başlıyorsa
    if current_sentence_id != prev_sentence_id:
        # Önceki cümlenin son context_window kelimesini al
        prev_sentence_tokens = [
            strip_punctuation(tokens[i]) for i in range(len(tokens)) 
            if i < current_position and sentence_ids[i] == prev_sentence_id
        ]
        return prev_sentence_tokens[-context_window:] if prev_sentence_tokens else []
    else:
        # Aynı cümle içindeyse, sadece son context_window kelimeyi al
        start = max(0, current_position - context_window)
        return [strip_punctuation(t) for t in tokens[start:current_position]]


# Test fonksiyonu
if __name__ == "__main__":
    test_text = """
    Bugün hava çok güzel! Ali okula gitti. 
    Öğretmen @ ders anlattı == test@email.com, matematik öğretti.
    Matematik ## çok zor... Ama çalışırsan başarırsın!
    Yarın: sınav var; hazırlan.
    """
    
    print("=" * 60)
    print("ORIJINAL:")
    print(test_text)
    print("=" * 60)
    
    cleaned = full_clean(test_text, lowercase=True, keep_punctuation=True)
    print("\nTEMİZLENMİŞ:")
    print(cleaned)
    print("=" * 60)
    
    sentences = split_sentences(cleaned)
    print("\nCÜMLELER:")
    for i, sent in enumerate(sentences):
        print(f"{i+1}. {sent}")
    print("=" * 60)
    
    tokens, sent_ids = tokenize_with_sentences(cleaned)
    print("\nTOKENS + SENTENCE IDs:")
    for token, sid in zip(tokens, sent_ids):
        is_ending = "📍" if is_sentence_ending(token) else "  "
        print(f"  {is_ending} {token:20} -> cümle {sid}")
    print("=" * 60)
    
    # Bağlam testi
    print("\nBAĞLAM TESTİ (Cross-sentence):")
    for i in range(len(tokens)):
        if i == 0 or (i > 0 and sent_ids[i] != sent_ids[i-1]):
            # Yeni cümle başlangıcı
            context = get_sentence_context(tokens, sent_ids, i, context_window=3)
            current_token = tokens[i] if i < len(tokens) else "END"
            print(f"\n🆕 Yeni cümle başlıyor: '{current_token}'")
            print(f"   Önceki cümle bağlamı: {context}")