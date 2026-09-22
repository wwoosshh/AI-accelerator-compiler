import torch
def fn(x):
    y = x.view(torch.int32) * 2
    y.sub_(-4)
    x[:, 2:5] = 2
    return y.view(torch.int64)
x = torch.arange(-12, 12, dtype=torch.int64, device="cuda").reshape(4, 6)
torch.compile(fn, backend="inductor")(x)
