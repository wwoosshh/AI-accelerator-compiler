import torch
def fn(t2, src):
    v2 = t2.reshape((24,))
    t2[:1, :8] = 5
    v2.index_add_(0, torch.tensor([5], device=t2.device), src[:1])
    return t2
t2 = torch.arange(-12, 12, dtype=torch.int64, device="cuda").reshape(3, 8); src = torch.full((2,), 100, dtype=torch.int64, device="cuda")
torch.compile(fn, backend="inductor", dynamic=True)(t2, src)
