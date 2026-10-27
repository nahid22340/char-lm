import torch

# 1. Load data
words = open('names.txt','r').read().splitlines()

# 2. Setup vocabulary mapping
chars = sorted(list(set((''.join(words)))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

# 3. Build the counting matrix N
N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

# 4. Normalize counts to probabilities (with +1 smoothing)
p = (N + 1).float()
p /= p.sum(1, keepdims=True)

# 5. Calculate Negative Log-Likelihood (Loss)
log_likelihood = 0.0
n = 0.0

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = p[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1

nll = -log_likelihood
print(f'{log_likelihood =}')
print(f'nll (Total Negative Log Likelihood): {nll:.4f}')
print(f'Average Loss (NLL / n): {nll/n:.4f}')

# 6. Generate Names
g = torch.Generator().manual_seed(2147483647)

print("\n ---Model Generated Names--- ")
for i in range(5):
    out = []
    ix = 0
    while True:
        p_dist = p[ix]
        ix = torch.multinomial(p_dist, num_samples=1, replacement=True, generator=g).item()
        if ix == 0:
            break
        out.append(itos[ix])
    print(''.join(out))