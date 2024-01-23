import torch
from naive_GPT import GPT, GPTConfig

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
head_size = 64
n_head = 6
n_layers = 6
dropout = 0.2
model_args = dict(n_layers=n_layers, head_size=head_size,
                  n_head=n_head, dropout=dropout,
                  vocab_size=vocab_size, n_embd=n_embd, device=device)

gptconfig = GPTConfig(**model_args)

m = GPT(gptconfig)
m.to(device)
m.load_state_dict(torch.load('weights/snaive.pt'))
m.eval()
context = torch.zeros((1,256), dtype=torch.long, device=device)
print(''.join(decode(m.generate(context, max_tokens=500)[0][-block_size:].tolist())))