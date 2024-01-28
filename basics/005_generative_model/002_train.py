import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm.auto import tqdm


###### CONFIG ######
SEED = 1337
split_ratio = 0.9

# Model parameters
block_size = 8          # chunk size
n_embd = 32             # embedding size
n_head_dim = 32         # attention head size

# Training parameters
batch_size = 4
eval_iters = 200
eval_interval = 1000
nu_epochs = 10000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
####################

torch.manual_seed(SEED)

# Read Dataset
def get_data():
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    print("length of dataset in characters: ", len(text))
    # print("first 1000 characters: ", text[:1000])
    return text


# Tokenization
def get_tokenizer(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # print(''.join(chars))
    # print(vocab_size)

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
    return encode, decode, vocab_size, chars


def random_chunks(data, block_size, batch_size):
    idxs = torch.randint(len(data) - block_size, (batch_size, ))
    return (
        torch.stack([data[i: i+block_size] for i in idxs]), 
        torch.stack([data[i+1: i+1+block_size] for i in idxs]),
    )


def calc_loss(model, data):
    losses = torch.zeros(eval_iters, device=device)
    for k in range(eval_iters):
        x, y = random_chunks(data, block_size, batch_size)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses[k] = loss.item()
    return losses.mean().item()


@torch.no_grad()
def loss_estimator(model, datasets):
    out = dict()
    model.eval()
    for name, data in datasets.items():
        loss = calc_loss(model, data)
        out[name] = loss
    model.train()
    return out


class AttentionHead(nn.Module):

    def __init__(self, n_embd, n_head_dim, block_size=block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, n_head_dim, bias=False)      # Query
        self.query = nn.Linear(n_embd, n_head_dim, bias=False)    # Key
        self.value = nn.Linear(n_embd, n_head_dim, bias=False)    # Value
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))  # Lower Triangular matrix
        print(self.mask.shape)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)                                      # B, T, C --> B, T, n_head_dim
        q = self.query(x)                                    # B, T, C --> B, T, n_head_dim
        v = self.value(x)                                    # B, T, C --> B, T, n_head_dim

        wei = q @ k.transpose(-2, -1) * C**-0.5              # B, T, n_head_dim @ B, n_head_dim, T --> B, T, T

        wei = wei.masked_fill(self.mask[:T, :T] == 0, float('-inf'))     # Mask out upper triangular matrix to -inf for softmax to return 0
        wei = F.softmax(wei, dim=-1)                        # Softmax along rows, gives out same wei as version 2.
        out = wei @ v                                     # B, T, T @ B, T, n_head_dim --> B, T, n_head_dim
        return out
    

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, block_size, n_embd, n_head_dim):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        self.sa_head = AttentionHead(n_embd=n_embd, n_head_dim=n_head_dim, block_size=block_size)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    @property
    def device(self):
        return self.tok_embedding.weight.device

    def forward(self, idx, targets=None):
        tok_emb = self.tok_embedding(idx) # B, T, C, (4, 8) --> (4, 8, 65)
        pos_emb = self.pos_embedding(torch.arange(idx.shape[-1], device=self.device)) # T, C, (8) --> (8, 65)
        x = tok_emb + pos_emb # B, T, C, (4, 8, 65) --> (4, 8, 65) B is broad casted.
        x = self.sa_head(x)
        logits = self.lm_head(x) # B, T, C, (4, 8, 65) --> (4, 8, 65)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        
        return logits, None
    
    def generate(self, idx, max_new_tokens, block_size=block_size):
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]         # Trim to block size
            logits, _ = self(idx_cond)              # B, T, C
            last_logits = logits[:, -1, :]          # B, -1, C --> B, C
           
            probs = F.softmax(last_logits, dim=-1)  # B, C
            next_idx = torch.multinomial(probs, num_samples=1) # B, 1
            idx = torch.cat([idx, next_idx], dim=1)
        return idx
    

# Read Dataset
text = get_data()
encode, decode, vocab_size, chars = get_tokenizer(text)
print("vocab size: ", vocab_size)

# tokenize text
data = torch.tensor(encode(text), dtype=torch.long)
print("data: ", data.shape, data.dtype)

# Train test split
n = int(len(data)*split_ratio)
train_data, val_data = data[:n], data[n:]
print("train and test split: ", train_data.shape, val_data.shape)

# Batch and chunk
for t in range(1, block_size+1):
    inputs = data[:t]
    targets = data[t]
    print(inputs, "->", targets)


# Init model and optimizer
m = BigramLanguageModel(
    vocab_size=vocab_size, 
    block_size=block_size, 
    n_embd=n_embd,
    n_head_dim=n_head_dim,
)
m = m.to(device)
print(m.device)
print(m)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# Training
print("Training")
for steps in tqdm(range(1, 1+nu_epochs), disable=True):

    x, y = random_chunks(data=train_data, block_size=block_size, batch_size=batch_size)
    x, y = x.to(device=device), y.to(device=device)
    logits, loss = m(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    # print(loss.item())

    if steps % eval_interval == 0 or steps == 1:
        out = loss_estimator(model=m, datasets={'train': train_data, 'val': val_data})
        print(f"Step : {steps:<8}| train_loss : {out['train']:.6f} | val_loss : {out['val']:.6f}")

out = loss_estimator(model=m, datasets={'train': train_data, 'val': val_data})
print(f"Final metrics train_loss : {out['train']:.6f} | val_loss : {out['val']:.6f}")

# Generate text
gen_text = decode(m.generate(
    idx=torch.zeros((1, 1), dtype=torch.long, device=device), 
    max_new_tokens=500
)[0].tolist())
print(f"Generated Text\n", '-'*100, '\n')
print(gen_text)
print('-'*100)
