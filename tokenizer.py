"""
Basit karakter-seviyesi tokenizer (Türkçe için optimize edilmiş)
"""
import json
import os
from collections import Counter
from typing import List, Dict


class SimpleTokenizer:
    """
    Karakter-seviyesi tokenizer.
    Küçük veri setleri için yeterli ve hızlı.
    """
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        
        # Özel tokenlar
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.BOS_TOKEN = "<BOS>"  # Begin of sentence
        self.EOS_TOKEN = "<EOS>"  # End of sentence
        
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        
        # Özel tokenları ekle
        self.char_to_id = {
            self.PAD_TOKEN: self.pad_id,
            self.UNK_TOKEN: self.unk_id,
            self.BOS_TOKEN: self.bos_id,
            self.EOS_TOKEN: self.eos_id,
        }
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
    
    def build_vocab(self, texts: List[str]):
        """
        Verilen metinlerden vocabulary oluşturur.
        
        Args:
            texts: Eğitim metinleri listesi
        """
        # Tüm karakterleri say
        char_counter = Counter()
        for text in texts:
            char_counter.update(text)
        
        # En sık kullanılan karakterleri al
        most_common = char_counter.most_common(self.vocab_size - 4)  # 4 özel token için yer bırak
        
        # Vocabulary'e ekle
        for char, _ in most_common:
            if char not in self.char_to_id:
                char_id = len(self.char_to_id)
                self.char_to_id[char] = char_id
                self.id_to_char[char_id] = char
        
        print(f"✓ Vocabulary oluşturuldu: {len(self.char_to_id)} karakter")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Metni token ID'lerine çevirir.
        
        Args:
            text: Encode edilecek metin
            add_special_tokens: BOS/EOS eklensin mi
            
        Returns:
            Token ID listesi
        """
        ids = []
        
        if add_special_tokens:
            ids.append(self.bos_id)
        
        for char in text:
            ids.append(self.char_to_id.get(char, self.unk_id))
        
        if add_special_tokens:
            ids.append(self.eos_id)
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Token ID'lerini metne çevirir.
        
        Args:
            ids: Token ID listesi
            skip_special_tokens: Özel tokenları atla
            
        Returns:
            Metin
        """
        chars = []
        special_ids = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        
        for id in ids:
            if skip_special_tokens and id in special_ids:
                continue
            chars.append(self.id_to_char.get(id, self.UNK_TOKEN))
        
        return "".join(chars)
    
    def save(self, filepath: str):
        """Tokenizer'ı kaydet."""
        data = {
            "vocab_size": self.vocab_size,
            "char_to_id": self.char_to_id,
            "id_to_char": {int(k): v for k, v in self.id_to_char.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✓ Tokenizer kaydedildi: {filepath}")
    
    def load(self, filepath: str):
        """Tokenizer'ı yükle."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.vocab_size = data["vocab_size"]
        self.char_to_id = data["char_to_id"]
        self.id_to_char = {int(k): v for k, v in data["id_to_char"].items()}
        print(f"✓ Tokenizer yüklendi: {filepath}")
    
    def __len__(self):
        return len(self.char_to_id)


# Test
if __name__ == "__main__":
    tokenizer = SimpleTokenizer(vocab_size=500)
    
    test_texts = [
        "Bugün hava çok güzel.",
        "Ali okula gitti.",
        "Matematik dersi çok zor.",
    ]
    
    tokenizer.build_vocab(test_texts)
    
    # Encode/Decode test
    text = "Bugün hava güzel."
    print(f"\nOrijinal: {text}")
    
    ids = tokenizer.encode(text)
    print(f"Encoded: {ids}")
    
    decoded = tokenizer.decode(ids)
    print(f"Decoded: {decoded}")
    
    # Save/Load test
    tokenizer.save("tokenizer_test.json")