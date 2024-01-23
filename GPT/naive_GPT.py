import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k_embed = nn.Linear(config.n_embd, config.head_size, bias=False)
        self.q_embed = nn.Linear(config.n_embd, config.head_size, bias=False)
        self.v_embed = nn.Linear(config.n_embd, config.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        k = self.k_embed(x) #(B, T, d)
        q = self.q_embed(x) #(B, T, d)
        v = self.v_embed(x) #(B, T, d)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril==0, float('-inf'))
        wei = torch.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        wei = wei @ v
        return wei

class FeedForward(nn.Module):
    def __init__(self, config):
      super().__init__()
      self.net = nn.Sequential(
         nn.Linear(config.n_embd, 4 * config.n_embd),
         nn.ReLU(),
         nn.Linear(4 * config.n_embd, config.n_embd),
         nn.Dropout(config.dropout)
      )
    
    def forward(self, x):
      return self.net(x);

class MultiHeadAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
    self.proj = nn.Linear(config.n_head * config.head_size, config.n_embd)
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, x):
    out = torch.concat([head(x) for head in self.heads], dim=-1)
    out = self.proj(out)
    out = self.dropout(out)
    return out

class Block(nn.Module):
    def __init__(self, config):
      super().__init__()
      self.sa_head = MultiHeadAttention(config)
      self.ffwd = FeedForward(config)
      self.ln1 = nn.LayerNorm(config.n_embd)
      self.ln2 = nn.LayerNorm(config.n_embd)
    def forward(self, x):
      x = x + self.sa_head(x)
      x = x + self.ffwd(x) 
      return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.block = nn.Sequential(*[Block(config) for _ in range(config.n_layers)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.config = config

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embd = self.token_embedding_table(idx) #(B,T,C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=self.config.device))
        x = tok_embd + pos_embd
        x = self.block(x)
        logits = self.lm_head(x) #(B, T, vocab_size)
        
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(T*B, C)
            targets = targets.view(T*B)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_tokens):
      for _ in range(max_tokens):
        idx_cond = idx[:, -self.config.block_size:]
        logits, loss = self.forward(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)

        idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)
      
      return idx

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 32
    head_size: int = 16
    n_layers: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    device: str = 'cpu'
    head_size: int = n_embd // n_head 


