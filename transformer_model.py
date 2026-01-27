"""
LSTM Tabanlı Dil Modeli
Karakter seviyesinde çalışır ve N-gram'dan çok daha iyi sonuç verir
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import random
from text_utils import full_clean, restore_punctuation_from_tokens


class CharDataset(Dataset):
    """Karakter bazlı dataset"""
    def __init__(self, text, seq_length=100):
        self.text = text
        self.seq_length = seq_length
        
        # Karakter sözlüğü oluştur
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
    def __len__(self):
        return len(self.text) - self.seq_length
    
    def __getitem__(self, idx):
        # Sequence ve target
        chunk = self.text[idx:idx + self.seq_length + 1]
        
        # Karakterleri index'e çevir
        input_seq = [self.char_to_idx[ch] for ch in chunk[:-1]]
        target_seq = [self.char_to_idx[ch] for ch in chunk[1:]]
        
        return torch.tensor(input_seq), torch.tensor(target_seq)


class LSTMLanguageModel(nn.Module):
    """LSTM Tabanlı Language Model"""
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super(LSTMLanguageModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.2)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        # Embedding
        embeds = self.embedding(x)
        
        # LSTM
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        # Output
        output = self.fc(lstm_out)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """Hidden state başlat"""
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))


class TransformerPredictor:
    """Transformer/LSTM Model Wrapper"""
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.dataset = None
        self.char_to_idx = None
        self.idx_to_char = None
        self.vocab_size = 0
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def train(self, text_file, seq_length=100, embedding_dim=128, 
              hidden_dim=256, num_layers=2, epochs=10, batch_size=64, 
              learning_rate=0.001, save_path='lstm_model.pth'):
        """Modeli eğit"""
        print(f"📚 Eğitim başlıyor...")
        print(f"   Device: {self.device}")
        
        # Veriyi yükle ve temizle
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"   Metin uzunluğu: {len(text):,} karakter")
        
        # Dataset oluştur
        self.dataset = CharDataset(text, seq_length)
        self.char_to_idx = self.dataset.char_to_idx
        self.idx_to_char = self.dataset.idx_to_char
        self.vocab_size = self.dataset.vocab_size
        
        print(f"   Vocabulary boyutu: {self.vocab_size}")
        print(f"   Sequence uzunluğu: {seq_length}")
        
        # DataLoader
        dataloader = DataLoader(self.dataset, batch_size=batch_size, 
                               shuffle=True, num_workers=0)
        
        # Model oluştur
        self.model = LSTMLanguageModel(
            self.vocab_size, embedding_dim, hidden_dim, num_layers
        ).to(self.device)
        
        # Loss ve optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Eğitim döngüsü
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward
                optimizer.zero_grad()
                outputs, _ = self.model(inputs)
                
                # Loss hesapla (reshape gerekli)
                loss = criterion(outputs.reshape(-1, self.vocab_size), 
                               targets.reshape(-1))
                
                # Backward
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Progress
                if batch_idx % 50 == 0:
                    print(f"   Epoch {epoch+1}/{epochs} - Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(dataloader)
            print(f"✅ Epoch {epoch+1}/{epochs} tamamlandı - Avg Loss: {avg_loss:.4f}")
        
        # Modeli kaydet
        self.save_model(save_path)
        print(f"✅ Model kaydedildi: {save_path}")
    
    def save_model(self, path):
        """Modeli kaydet"""
        torch.save({
            'model_state': self.model.state_dict(),
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocab_size': self.vocab_size,
            'hidden_dim': self.model.hidden_dim,
            'num_layers': self.model.num_layers,
            'embedding_dim': self.model.embedding.embedding_dim
        }, path)
    
    def load_model(self, path):
        """Modeli yükle"""
        print(f"📦 LSTM Model yükleniyor: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.char_to_idx = checkpoint['char_to_idx']
        self.idx_to_char = checkpoint['idx_to_char']
        self.vocab_size = checkpoint['vocab_size']
        
        # Model oluştur
        self.model = LSTMLanguageModel(
            self.vocab_size,
            checkpoint['embedding_dim'],
            checkpoint['hidden_dim'],
            checkpoint['num_layers']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        
        print(f"✅ LSTM Model yüklendi - Vocab: {self.vocab_size}")
    
    def generate(self, start_text, max_length=200, temperature=0.8):
        """Metin üret"""
        if self.model is None:
            return ""
        
        self.model.eval()
        
        with torch.no_grad():
            # Başlangıç metnini encode et
            current_text = start_text
            hidden = None
            
            for _ in range(max_length):
                # Son karakteri al
                if current_text:
                    last_char = current_text[-1]
                    if last_char not in self.char_to_idx:
                        break
                    
                    char_idx = self.char_to_idx[last_char]
                    input_tensor = torch.tensor([[char_idx]]).to(self.device)
                    
                    # Predict
                    output, hidden = self.model(input_tensor, hidden)
                    
                    # Temperature sampling
                    output = output[0, -1] / temperature
                    probs = torch.softmax(output, dim=0)
                    
                    # Sample
                    next_idx = torch.multinomial(probs, 1).item()
                    next_char = self.idx_to_char[next_idx]
                    
                    current_text += next_char
                    
                    # Cümle bitirici token kontrolü
                    if next_char in ['.', '!', '?'] or 'TR001' in current_text[-10:] or 'TR003' in current_text[-10:] or 'TR004' in current_text[-10:]:
                        break
                else:
                    break
            
            # Sadece eklenen kısmı dön
            generated = current_text[len(start_text):]
            return generated
    
    def predict(self, text, use_tokens=True):
        """Kelime/cümle tamamlama"""
        if self.model is None:
            return ""
        
        # Metni temizle
        text_clean = full_clean(text, lowercase=True, use_tokens=use_tokens)
        
        # Metin üret
        generated = self.generate(text_clean, max_length=100, temperature=0.7)
        
        # Tokenları restore et
        if use_tokens:
            generated = restore_punctuation_from_tokens(generated)
        
        return generated


# Test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        # Eğitim modu
        predictor = TransformerPredictor()
        predictor.train(
            text_file='story_tokenized_temp.txt',
            seq_length=100,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
            epochs=5,
            batch_size=64,
            learning_rate=0.001,
            save_path='lstm_model.pth'
        )
    else:
        # Test modu
        if os.path.exists('lstm_model.pth'):
            predictor = TransformerPredictor('lstm_model.pth')
            
            test_texts = [
                "bugün hava",
                "türkiye",
                "futbol maçı"
            ]
            
            print("\n🧪 Test Sonuçları:\n")
            for text in test_texts:
                result = predictor.predict(text)
                print(f"Input: {text}")
                print(f"Output: {result}\n")
        else:
            print("❌ lstm_model.pth bulunamadı!")
            print("Önce modeli eğitin: python transformer_model.py train")
