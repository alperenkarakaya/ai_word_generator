"""
LSTM Model Eğitim Scripti
"""
import sys
import os

# PyTorch kontrolü
try:
    import torch
    print(f"✅ PyTorch {torch.__version__} kurulu")
    print(f"   CUDA: {'Evet' if torch.cuda.is_available() else 'Hayır (CPU kullanılacak)'}")
except ImportError:
    print("❌ PyTorch yüklü değil!")
    print("\nKurmak için:")
    print("  pip install torch")
    sys.exit(1)

from transformer_model import TransformerPredictor

def train_lstm_model():
    """LSTM modelini eğit"""
    
    # Eğitim dosyası kontrolü
    if not os.path.exists('story_tokenized_temp.txt'):
        print("❌ story_tokenized_temp.txt bulunamadı!")
        print("Önce 'python create_pickle.py' çalıştırın")
        return
    
    # Model oluştur ve eğit
    predictor = TransformerPredictor()
    
    print("\n🚀 LSTM Model Eğitimi Başlıyor...")
    print("=" * 60)
    print("Hyperparameters:")
    print("  - Sequence Length: 100")
    print("  - Embedding Dim: 128")
    print("  - Hidden Dim: 256")
    print("  - Num Layers: 2")
    print("  - Epochs: 5 (daha fazla için kod içinde değiştirin)")
    print("  - Batch Size: 64")
    print("  - Learning Rate: 0.001")
    print("=" * 60)
    print()
    
    predictor.train(
        text_file='story_tokenized_temp.txt',
        seq_length=100,        # Sequence uzunluğu
        embedding_dim=128,     # Embedding boyutu
        hidden_dim=256,        # Hidden layer boyutu
        num_layers=2,          # LSTM layer sayısı
        epochs=5,              # Epoch sayısı (artırabilirsiniz)
        batch_size=64,         # Batch size
        learning_rate=0.001,   # Learning rate
        save_path='lstm_model.pth'
    )
    
    print("\n✅ Eğitim tamamlandı!")
    print("\nTest etmek için:")
    print("  python train_lstm.py test")
    print("\nFlask ile kullanmak için:")
    print("  python app.py")
    print("  (LSTM otomatik yüklenecek)")


def test_lstm_model():
    """LSTM modelini test et"""
    
    if not os.path.exists('lstm_model.pth'):
        print("❌ lstm_model.pth bulunamadı!")
        print("Önce modeli eğitin: python train_lstm.py")
        return
    
    predictor = TransformerPredictor('lstm_model.pth')
    
    test_texts = [
        "bugün hava",
        "türkiye",
        "futbol maçı",
        "bir zamanlar"
    ]
    
    print("\n🧪 LSTM Model Test Sonuçları:")
    print("=" * 60)
    
    for text in test_texts:
        result = predictor.predict(text, use_tokens=True)
        print(f"\nInput:  {text}")
        print(f"Output: {text}{result}")
    
    print("\n" + "=" * 60)
    print("✅ Test tamamlandı!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_lstm_model()
    else:
        train_lstm_model()
