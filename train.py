
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AdamW, get_scheduler
import tiktoken
from dataclasses import dataclass

class DataLoaderLite:
    def __init__(self, file_path, B, T):
        self.B = B
        self.T = T
        
        # Load tokens from disk and store in memory
        with open(file_path, 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'Loaded {len(self.tokens)} tokens.')

        # State
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # Inputs
        y = (buf[1:]).view(B, T)  # Targets
        
        # Advance position
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            config.n_embd, 
            config.n_head, 
            dropout=config.dropout, 
            batch_first=True,
            bias=True
        )
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=1e-6)
        self.dropout = nn.Dropout(config.dropout)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=True),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=True),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        residual_scale = 0.7
        attn_output = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x))[0]
        x = x + residual_scale * self.dropout(attn_output)
        x = x + residual_scale * self.dropout(self.mlp(self.ln_2(x)))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}."

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), label_smoothing=0.05)
        return logits, loss

# Load model
config = GPTConfig()
model = GPT(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load data
file_path = "input.txt"
train_loader = DataLoaderLite(file_path, B=32, T=128)

# Optimizer and Scheduler
optimizer = AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1
)
num_training_steps = 10000
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=1200,
    num_training_steps=num_training_steps,
    num_cycles=4
)

# Training loop
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
accumulation_steps = 16

# Add memory optimization before training loop
torch.cuda.empty_cache()
if torch.cuda.is_available():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
# Add exponential moving average for loss tracking
ema_loss = None
ema_alpha = 0.95

# Add gradient clipping threshold
max_grad_norm = 0.5

# Add learning rate warmup function
def get_lr(step, num_warmup_steps, max_lr):
    if step < num_warmup_steps:
        return max_lr * (step / num_warmup_steps)
    return max_lr

# Add before training loop
best_loss = float('inf')
save_dir = 'model_checkpoints'
os.makedirs(save_dir, exist_ok=True)

for step in tqdm(range(num_training_steps), desc="Training"):
    # Learning rate warmup
    if step < 1200:  # warmup steps
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(step, 1200, 3e-4)
    
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)

    if step % accumulation_steps == 0:
        optimizer.zero_grad(set_to_none=True)

    with torch.cuda.amp.autocast(enabled=(scaler is not None)):
        logits, loss = model(x, y)
        loss = loss / accumulation_steps

    # Update EMA loss
    if ema_loss is None:
        ema_loss = loss.item()
    else:
        ema_loss = ema_loss * ema_alpha + loss.item() * (1 - ema_alpha)

    if scaler:
        scaler.scale(loss).backward()
        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
    else:
        loss.backward()
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            lr_scheduler.step()

    if step % 100 == 0:
        print(f"Step {step + 1}/{num_training_steps}, Loss: {ema_loss:.6f}")
        # Save checkpoint if loss improved
        if ema_loss < best_loss:
            best_loss = ema_loss
            checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': ema_loss,
                'config': config,
            }
            torch.save(checkpoint, f'{save_dir}/best_model.pt')
            print(f"Saved best model with loss: {ema_loss:.6f}")
        
        # Save periodic checkpoint
        if step % 1000 == 0:
            checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': ema_loss,
                'config': config,
            }
            torch.save(checkpoint, f'{save_dir}/checkpoint_{step}.pt')
            print(f"Saved checkpoint at step {step}")

# Save final model
final_checkpoint = {
    'step': num_training_steps,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': ema_loss,
    'config': config,
}
torch.save(final_checkpoint, f'{save_dir}/final_model.pt')

print(f"Training completed. Final loss: {ema_loss:.6f}")
print(f"Best loss achieved: {best_loss:.6f}")
print(f"Models saved in {save_dir}")
print("Note: Download your models from the Kaggle output tab")
