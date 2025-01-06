import torch
import torch.nn as nn
from torch.nn import functional as F


batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
#--------------------------------

torch.manual_seed(1337)

with open("../input.txt") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for s,i in stoi.items()}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda i: ''.join([itos[num] for num in i])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split=="train" else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x,y



class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, target=None):
        logit = self.token_embedding_table(idx) #(B ie batch,T ie time,Channel ie vocab_size)

        if target is None:
            loss = None
        else:
            B,T,C = logit.shape
            logit = logit.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logit, target)
        return logit, loss

    def generate(self, idx, max_num_tokens=1):
        for _ in range(max_num_tokens):
            logit, _ = self(idx)
            logit = logit[:, -1, :]#take logit of the last time dimension (B,C)
            prob = F.softmax(logit, dim=-1)
            idx_next = torch.multinomial(prob, num_samples=1) #(B,1) each batch predicts a token
            idx = torch.cat((idx, idx_next), dim=1) #
        return idx

@torch.no_grad
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            _, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32

for steps in range(max_iters):

    if steps%eval_iters==0:
        losses = estimate_loss()
        print(f"step {steps} train loss:{losses['train']:.4f} val loss:{losses['val']:.4f}")
    xb, yb = get_batch("train")
    logit,loss = m(xb,yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, 100)[0].tolist()))