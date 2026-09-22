# -*- coding: utf-8 -*-
"""Family L trigger search: which copy form + which mutation makes Inductor copy AFTER the mutation?"""
import torch

def L1(dst, src): dst[0:, :] = src[0:, :]; src.add_(1); return src
def L2(dst, src): dst[:] = src[0:, :]; src.add_(1); return src
def L3(dst, src): dst[0:, :] = src; src.add_(1); return src
def L4(dst, src): dst[:, :] = src[:, :]; src.add_(1); return src
def L5(dst, src): dst[1:, :] = src[1:, :]; src.add_(1); return src
def L6(dst, src): dst[0:, :] = src[0:, :].clone(); src.add_(1); return src
def L7(dst, src): dst[0:, :] = src[0:, :]; src.add_(1)              # no return
def L8(dst, src): dst[0:, :] = src[0:, :]; src.add_(1); return dst
def L9(dst, src): dst[0:, :] = src[0:, :]; return src.add_(1)
def L10(dst, src): dst[0:, :] = src[0:, :]; src.add_(1); return dst.sum()
def L11(dst, src): dst.copy_(src[0:, :]); src.add_(1); return src
def L12(dst, src): dst[0:, :].copy_(src); src.add_(1); return src
def L13(dst, src): dst[0:, :] = src[0:, :]; src.index_fill_(0, torch.tensor([0], device=src.device), -2.0); return src
def L14(dst, src): dst[:] = src; src.index_fill_(0, torch.tensor([0], device=src.device), -2.0); return src

def check(fn, dtype=torch.float32):
    res = []
    for compiled in (False, True):
        torch.manual_seed(0)
        dst = torch.zeros(3, 4, device="cuda", dtype=dtype); src = torch.randn(3, 4, device="cuda").to(dtype)
        torch._dynamo.reset()
        (torch.compile(fn, backend="inductor") if compiled else fn)(dst, src)
        res.append((dst.clone(), src.clone()))
    ok = torch.equal(res[0][0], res[1][0]) and torch.equal(res[0][1], res[1][1])
    return ok, res

if __name__ == "__main__":
    import inspect
    print("torch", torch.__version__)
    for fn in (L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12, L13, L14):
        ok, res = check(fn)
        body = inspect.getsource(fn).split(":", 1)[1].strip()
        print(f"{'OK ' if ok else 'BUG'} {fn.__name__:4s} {body:75s}" + ("" if ok else f" | dst[0,0] eager {res[0][0][0,0].item():+.3f} compiled {res[1][0][0,0].item():+.3f}"))
