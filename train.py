"""
Transformer model eğitim scripti
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import time
from tqdm import tqdm

from tokenizer import SimpleTokenizer
from transformer_model import TurkishGPT, TransformerConfig
from text_utils import full_clean


class TextDataset(Dataset):
    """Metin dataset'i."""
    
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Metinleri encode et
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=True)
            
            # Uzun metinleri parçalara böl
            for i in range(0, len(ids) - max_length, max_length // 2):
                chunk = ids[i:i + max_length]
                if len(chunk) == max_length:
                    self.examples.append(chunk)
        
        print(f"✓ Dataset oluşturuldu: {len(self.examples)} örnek")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ids = self.examples[idx]
        
        # Input: tüm tokenlar (son hariç)
        # Target: tüm tokenlar (ilk hariç)
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)
        
        return input_ids, target_ids


class Trainer:
    """Model eğitim sınıfı."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        learning_rate=3e-4,
        weight_decay=0.01,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir='checkpoints',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 10,  # 10 epoch için
            eta_min=learning_rate * 0.1
        )
        
        # Checkpoint directory oluştur
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        print(f"✓ Trainer hazır (device: {device})")
    
    def train_epoch(self):
        """Bir epoch eğitim."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (input_ids, target_ids) in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Forward pass
            logits, loss = self.model(input_ids, targets=target_ids)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Stats
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Validation."""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for input_ids, target_ids in tqdm(self.val_loader, desc="Validation"):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            logits, loss = self.model(input_ids, targets=target_ids)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, filename, is_best=False):
        """Checkpoint kaydet."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.model.config.__dict__,
        }
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"✓ Checkpoint kaydedildi: {filepath}")
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"✓ En iyi model güncellendi: {best_path}")
    
    def load_checkpoint(self, filename):
        """Checkpoint yükle."""
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"⚠ Checkpoint bulunamadı: {filepath}")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"✓ Checkpoint yüklendi: {filepath}")
        return True
    
    def train(self, num_epochs, save_every=1, validate_every=1):
        """Ana eğitim loop'u."""
        print(f"\n{'='*60}")
        print(f"🚀 Eğitim başlıyor: {num_epochs} epoch")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            print(f"\nEpoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}")
            
            # Validate
            if self.val_loader is not None and (epoch + 1) % validate_every == 0:
                val_loss = self.validate()
                print(f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', is_best=True)
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
        
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✓ Eğitim tamamlandı! Süre: {elapsed / 60:.2f} dakika")
        print(f"{'='*60}\n")


def prepare_data(data_file='story.txt', train_split=0.9):
    """Veri hazırlama."""
    print("📊 Veri hazırlanıyor...\n")
    
    # Metni yükle
    if not os.path.exists(data_file):
        print(f"❌ Veri dosyası bulunamadı: {data_file}")
        return None, None
    
    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Temizle
    text_clean = full_clean(text, lowercase=True, keep_punctuation=True)
    
    # Cümlelere böl
    from text_utils import split_sentences
    sentences = split_sentences(text_clean)
    
    print(f"✓ {len(sentences)} cümle bulundu")
    
    # Train/val split
    split_idx = int(len(sentences) * train_split)
    train_texts = sentences[:split_idx]
    val_texts = sentences[split_idx:]
    
    print(f"  Train: {len(train_texts)} cümle")
    print(f"  Val: {len(val_texts)} cümle\n")
    
    return train_texts, val_texts


def main():
    """Ana eğitim fonksiyonu."""
    print("\n" + "="*60)
    print("🧠 TURKISH GPT EĞİTİM")
    print("="*60 + "\n")
    
    # Hyperparameters
    VOCAB_SIZE = 500
    D_MODEL = 256
    N_HEADS = 4
    N_LAYERS = 3
    D_FF = 1024
    MAX_SEQ_LENGTH = 128
    DROPOUT = 0.1
    
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 20
    TRAIN_SPLIT = 0.9
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # 1. Veri hazırla
    train_texts, val_texts = prepare_data('story.txt', train_split=TRAIN_SPLIT)
    
    if train_texts is None:
        print("❌ Veri hazırlanamadı!")
        return
    
    # 2. Tokenizer oluştur
    print("🔤 Tokenizer oluşturuluyor...\n")
    tokenizer = SimpleTokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.build_vocab(train_texts + val_texts)
    tokenizer.save('tokenizer.json')
    print()
    
    # 3. Dataset oluştur
    print("📦 Dataset oluşturuluyor...\n")
    train_dataset = TextDataset(train_texts, tokenizer, max_length=MAX_SEQ_LENGTH)
    val_dataset = TextDataset(val_texts, tokenizer, max_length=MAX_SEQ_LENGTH)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    print()
    
    # 4. Model oluştur
    print("🏗️  Model oluşturuluyor...\n")
    config = TransformerConfig(
        vocab_size=len(tokenizer),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_length=MAX_SEQ_LENGTH,
        dropout=DROPOUT,
    )
    
    model = TurkishGPT(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model oluşturuldu")
    print(f"  Parametreler: {num_params:,}")
    print(f"  Model boyutu: ~{num_params * 4 / 1024 / 1024:.2f} MB\n")
    
    # 5. Trainer oluştur
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=LEARNING_RATE,
        device=device,
        checkpoint_dir='checkpoints',
    )
    print()
    
    # 6. Eğitim başlat
    trainer.train(
        num_epochs=NUM_EPOCHS,
        save_every=5,
        validate_every=1,
    )
    
    # 7. Test generation
    print("\n" + "="*60)
    print("🧪 Test Generation")
    print("="*60 + "\n")
    
    model.eval()
    test_prompt = "bugün hava"
    
    input_ids = torch.tensor([tokenizer.encode(test_prompt, add_special_tokens=True)]).to(device)
    
    generated_ids = model.generate(
        input_ids,
        max_new_tokens=50,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
    )
    
    generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())
    
    print(f"Prompt: {test_prompt}")
    print(f"Generated: {generated_text}\n")
    
    print("✅ Eğitim tamamlandı!")


if __name__ == "__main__":
    main()