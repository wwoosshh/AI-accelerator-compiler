import torch
def idx(x, i): return torch.tensor([i], device=x.device)
def c_add_(x): x.add_(5); x.index_fill_(1, idx(x, 2), -1); x[x > 0] = 2; return x
x = torch.arange(1, 13, dtype=torch.int64, device="cuda").reshape(3, 4)
torch.compile(c_add_, backend="inductor")(x)
