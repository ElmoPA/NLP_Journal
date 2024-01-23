from tqdm import tqdm
import torch
import torch.nn as nn
from naive_GPT import GPTConfig, GPT
torch.manual_seed(1337)

with open('notebook/input.txt', encoding='utf-8') as f:
  text = f.read()
char = sorted(list(set(text)))
vocab_size = len(char)

stoi = {ch: i for i, ch in enumerate(char)}
itos = {i: ch for i, ch in enumerate(char)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: [itos[i] for i in l]
n = int(len(text) * 0.9)
train_data = torch.tensor(encode(text[:n]), dtype=torch.long)
val_data = torch.tensor(encode(text[n:]), dtype=torch.long)

batch_size = 64 
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
head_size = 16
n_head = 6
n_layers = 6
dropout = 0.2
model_args = dict(n_layers=n_layers, head_size=head_size,
                  n_head=n_head, dropout=dropout,
                  vocab_size=vocab_size, n_embd=n_embd, device=device)
gptconfig = GPTConfig(**model_args)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)- block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

m = GPT(gptconfig)
if torch.cuda.device_count() > 1:
      m = nn.DataParallel(m)
m = m.to(device)
  
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
pbar = tqdm(range(max_iters), desc="Training in progress")

for step in pbar:
  x, y = get_batch('train')
  vx, vy = get_batch('val')
  x,y = x.to(device), y.to(device)
  vx, vy = vx.to(device), vy.to(device)
  logits, loss = m(x, y)
  loss = loss.mean()
  _, vloss = m(vx, vy)
  vloss = vloss.mean()
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
  if (step + 1) % 100 == 0:
    torch.save(m.state_dict(), 'path.pt')
  # break
  if step % 1 == 0:
      pbar.set_postfix(Step=step, Loss=loss.item(), Val_Loss=vloss.item())