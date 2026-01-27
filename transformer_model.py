"""
Mini Transformer modeli (GPT-style, decoder-only)
Türkçe metin üretimi için optimize edilmiş
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerConfig:
    """Transformer model konfigürasyonu."""
    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 1024,
        max_seq_length: int = 512,
        dropout: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_length = max_seq_length
        self.dropout = dropout


class PositionalEncoding(nn.Module):
    """Sinüzoidal pozisyonel encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Pozisyon encodinglari hesapla
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len, seq_len) - causal mask
        Returns:
            (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections ve head'lere böl
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        
        # Concat heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.W_o(out)
        
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Tek bir Transformer decoder bloğu."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config.d_model, config.n_heads, config.dropout)
        self.feed_forward = FeedForward(config.d_model, config.d_ff, config.dropout)
        
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: Causal mask
        Returns:
            (batch_size, seq_len, d_model)
        """
        # Self-attention + residual
        attn_out = self.attention(self.ln1(x), mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward + residual
        ff_out = self.feed_forward(self.ln2(x))
        x = x + self.dropout(ff_out)
        
        return x


class TurkishGPT(nn.Module):
    """
    Mini GPT-style Transformer model.
    Türkçe metin üretimi için.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_length)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output projection
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Xavier/Kaiming initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _create_causal_mask(self, seq_len, device):
        """
        Causal mask oluştur (sonraki tokenlara bakamaz).
        
        Returns:
            (1, 1, seq_len, seq_len)
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, input_ids, targets=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            targets: (batch_size, seq_len) - eğitim için
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: scalar (eğer targets verilmişse)
        """
        batch_size, seq_len = input_ids.size()
        
        # Embeddings
        x = self.token_embedding(input_ids)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Causal mask
        mask = self._create_causal_mask(seq_len, input_ids.device)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        # Loss hesapla (eğer targets varsa)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0  # PAD token
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=50,
        temperature=1.0,
        top_k=None,
        top_p=None,
    ):
        """
        Metin üret (autoregressive).
        
        Args:
            input_ids: (batch_size, seq_len) - başlangıç tokenları
            max_new_tokens: Kaç token üretilecek
            temperature: Sampling sıcaklığı
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            (batch_size, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Model forward
            logits, _ = self.forward(input_ids)
            
            # Sadece son tokenin logit'lerini al
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # İlk top_p'ye ulaşan indexleri bul
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


# Test
if __name__ == "__main__":
    print("🧠 Transformer Model Test\n")
    
    config = TransformerConfig(
        vocab_size=1000,
        d_model=256,
        n_heads=4,
        n_layers=3,
        d_ff=1024,
        max_seq_length=128,
        dropout=0.1,
    )
    
    model = TurkishGPT(config)
    print(f"✓ Model oluşturuldu")
    print(f"  Parametreler: {sum(p.numel() for p in model.parameters()):,}")
    
    # Dummy input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass
    logits, loss = model(input_ids, targets=input_ids)
    print(f"\n✓ Forward pass başarılı")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item() if loss is not None else 'N/A'}")
    
    # Generation test
    generated = model.generate(input_ids[:1], max_new_tokens=20, temperature=0.8, top_k=40)
    print(f"\n✓ Generation başarılı")
    print(f"  Generated shape: {generated.shape}")