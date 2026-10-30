import torch.nn.functional as F

xs, ys, zs = [], [], []
for w in words[:1]:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    ix3 = stoi[ch3]
    xs.append(ix1)
    ys.append(ix2)
    zs.append(ix3)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
zs = torch.tensor(zs)

# 1. One-hot encode both context characters
xenc1 = F.one_hot(xs, num_classes=27).float()
xenc2 = F.one_hot(ys, num_classes=27).float()

# 2. Concatenate them along dimension 1 (columns)
xenc = torch.cat([xenc1, xenc2], dim=1)  # Shape becomes (N, 54)

g = torch.Generator().manual_seed(2147483647)
w = torch.randn((54, 27), generator=g, requires_grad=True)

for k in range (44):
#   Forward Pass
  xenc = torch.cat([xenc1, xenc2], dim=1)  # Shape becomes (N, 54)
  logits = xenc @ w
  counts = logits.exp()
  probs = counts / counts.sum(1, keepdims = True)
  loss = -probs[torch.arange(len(zs)), zs].log().mean()

  # Backward Pass
  w.grad = None
  loss.backward()

  w.data += -0.1*w.grad
print(loss)

print("\n --- Model Generated Names (using Neural Net) --- ")
for i in range(5):
  out = []
  ix1 = stoi['.']  # Start token
  ix2 = stoi['.']  # Start token

  while True:
    # 1. One-hot encode current context characters
    xenc1 = F.one_hot(torch.tensor([ix1]), num_classes=27).float()
    xenc2 = F.one_hot(torch.tensor([ix2]), num_classes=27).float()
    xenc = torch.cat([xenc1, xenc2], dim=1) # Shape (1, 54)

    # 2. Forward pass through trained weights w
    logits = xenc @ w # Shape (1, 27)
    counts = logits.exp()
    p_dist = counts / counts.sum(1, keepdims=True) # Probabilities

    # 3. Sample the next character
    ix3 = torch.multinomial(p_dist[0], num_samples=1, replacement=True, generator=g).item()

    # 4. Stop if we hit the end token '.'
    if ix3 == stoi['.']:
      break

    out.append(itos[ix3])

    # 5. Shift context forward for the next step
    ix1 = ix2
    ix2 = ix3

  print(''.join(out))