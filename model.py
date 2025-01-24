import os
import math
import time
import inspect
from dataclasses import dataclass
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import AdamW, get_scheduler

class Block(nn.Module):
    def __init__(self, config):
        """
        Transformer Block

        Args:
            embed_dim: Dimensionality of the input embeddings
            num_heads: Number of attention heads
            ff_dim: Dimensionality of the feedforward network
            dropout: Dropout rate
        """
        super(Block, self).__init__()

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(config.n_embd, config.n_head, dropout=config.dropout, batch_first=True, bias=True)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # or GELU
            nn.Linear(4 * config.n_embd, config.n_embd)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(config.n_embd, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.n_embd, eps=1e-6)

        # Dropout
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        """
        Forward pass for the Transformer Block.

        Args:
            x: Input tensor of shape (seq_len, batch_size, embed_dim)
            mask: Attention mask (optional)

        Returns:
            Output tensor of shape (seq_len, batch_size, embed_dim)
        """
        # Multi-head self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feedforward network with residual connection
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)

        return x


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 8 # number of heads
    n_embd: int = 768 # embedding dimension
    dropout: float = 0.1



class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        # weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss