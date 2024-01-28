import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm.auto import tqdm


###### CONFIG ######
SEED = 1337
split_ratio = 0.9
block_size = 8  # chunk size
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


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    @property
    def device(self):
        return self.embedding.weight.device

    def forward(self, idx, targets=None):
        logits = self.embedding(idx) # B, T, C, (4, 8) --> (4, 8, 65)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        
        return logits, None
    
    def generate(self, idx, max_new_tokens):
        
        for _ in range(max_new_tokens):
            logits, _ = self(idx) # B, T, C
            last_logits = logits[:, -1, :]  # B, -1, C --> B, C
           
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
m = BigramLanguageModel(vocab_size=vocab_size)
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
