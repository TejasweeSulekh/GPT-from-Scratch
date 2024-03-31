import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
#----------------

torch.manual_seed(1337)

# reading it for inspecting
with open('D:\Stuff\VScode\Pytorch_Stuff\GPT\input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
# Here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# creating a mapping from characters to integers
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # Encoder takes a string and outputs a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # Decoder: take a list of integers, output a string

# Splitting the data in train test split
data = torch.tensor(encode(text), dtype = torch.long, device=device)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # generates a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    # here ix is sampled batch_size times randomly from len(data) - block_size
    # That is we are creating the initialization of the batches
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # (B,T,C)
        self.postition_embeding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size) 
        
    def __call__(self, idx, targets=None):
        B, T = idx.shape
        
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.postition_embeding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            # Pytorch expects the dimensions for negative log likelihood to be (B, C, T)
            # batch, channel, time to take care of this we squeeze the time into batches
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T) 
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples = 1) # (B, 1)
            # append samped index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
    
model = BigramLanguageModel()
m = model.to(device)

# crate a Pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)

# Training loop
for iters in range(max_iters):
    
    # every once in a while evaluate the loss on train and val sets
    if iters % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    
    # sample a batch of data
    xb, yb = get_batch('train')
    
    # evaluate the loss
    
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
# generate from the model
# context = torch.zeros((1,1), dtype = torch.long, device = device)
# print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))