import torch
from torch.func import functionalize
from torch.fx.experimental.proxy_tensor import make_fx
print("torch", torch.__version__)

def f(x):
    v = x.unfold(1, 1, 5)
    v.add_(v)
    return x

x = torch.arange(1, 25, dtype=torch.int64).reshape(4, 6)
ref = f(x.clone())
out = functionalize(f)(x.clone())
print("functionalize alone reproduces:", not torch.equal(ref, out))
print("  eager :", ref.flatten().tolist())
print("  func  :", out.flatten().tolist())
g = make_fx(functionalize(f, remove="mutations_and_views"))(x.clone())
print("--- functionalized graph ---")
print(g.code)
