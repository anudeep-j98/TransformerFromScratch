import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int = 50257
    max_seq_len: int = 2048
    dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_head = config.num_heads
        self.n_embd = config.dim
        
        # Linear projections for Q, K, V
        self.c_attn = nn.Linear(config.dim, 3 * config.dim) # [n_embd, 3 * n_embd]
        self.c_proj = nn.Linear(config.dim, config.dim) # [n_embd, n_embd]
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size() # [B, T, n_embd]
        
        # Linear projection and split into Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2) # [B, T, n_embd] each
        
        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, n_head, T, n_embd/n_head]
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, n_head, T, n_embd/n_head]
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, n_head, T, n_embd/n_head]
        
        # Attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5)) # [B, n_head, T, T]
        att = F.softmax(att, dim=-1) # [B, n_head, T, T]
        att = self.attn_dropout(att) # [B, n_head, T, T]
        
        # Weighted sum of values
        y = att @ v # [B, n_head, T, n_embd/n_head]
        
        # Reshape and project
        y = y.transpose(1, 2).contiguous().view(B, T, C) # [B, T, n_embd]
        y = self.c_proj(y) # [B, T, n_embd]
        y = self.resid_dropout(y) # [B, T, n_embd]
        
        return y

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.dim, 4 * config.dim) # [n_embd, 4 * n_embd]
        self.c_proj = nn.Linear(4 * config.dim, config.dim) # [4 * n_embd, n_embd]
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x) # [B, T, 4 * n_embd]
        x = F.gelu(x) # [B, T, 4 * n_embd]
        x = self.c_proj(x) # [B, T, n_embd]
        x = self.dropout(x) # [B, T, n_embd]
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.dim) # [n_embd]
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.dim) # [n_embd]
        self.mlp = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # [B, T, n_embd]
        x = x + self.mlp(self.ln_2(x)) # [B, T, n_embd]
        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.dim) # [vocab_size, n_embd]
        self.wpe = nn.Embedding(config.max_seq_len, config.dim) # [max_seq_len, n_embd]
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.dim) # [n_embd]
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False) # [n_embd, vocab_size]
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx):
        B, T = idx.size() # [B, T]
        
        # Positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0) # [1, T]
        
        # Token and position embeddings
        tok_emb = self.wte(idx) # [B, T, n_embd]
        pos_emb = self.wpe(pos) # [1, T, n_embd]
        
        # Combine embeddings and apply dropout
        x = self.drop(tok_emb + pos_emb) # [B, T, n_embd]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x) # [B, T, n_embd]
        
        # Final layer norm and linear projection
        x = self.ln_f(x) # [B, T, n_embd]
        logits = self.lm_head(x) # [B, T, vocab_size]
        
        return logits

if __name__ == '__main__':
    config = Config()
    model = DecoderOnlyTransformer(config)
    
    # Example usage
    batch_size = 4
    seq_len = 128
    
    # Generate random input
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits = model(input_ids)
    
    print("Input shape:", input_ids.shape)
    print("Output shape:", logits.shape) 