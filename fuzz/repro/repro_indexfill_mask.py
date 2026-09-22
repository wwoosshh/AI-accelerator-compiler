# Bug candidate D: index_fill_ followed by a boolean-mask assignment on the same tensor gives
# wrong results under torch.compile (Inductor): the mask appears to be evaluated on the value
# *before* index_fill_ was applied (mutation ordering).
import sys, torch
print("torch", torch.__version__)

def f_min(x):                                  # minimal
    x.index_fill_(1, torch.tensor([0], device=x.device), -1)
    x[x > 0] = 2
    return x

def f_fuzz(x):                                 # as found by the fuzzer
    x[:, :] += 5
    x.index_fill_(1, torch.tensor([2], device=x.device), -1)
    x[x > 0] = 2
    return x

def f_slice(x):                                # variant: slice assignment instead of index_fill_
    x[:, 0] = -1
    x[x > 0] = 2
    return x

def f_masked_fill(x):                          # variant: masked_fill_ instead of index_put_
    x.index_fill_(1, torch.tensor([0], device=x.device), -1)
    x.masked_fill_(x > 0, 2)
    return x

def f_where(x):                                # variant: functional where after index_fill_
    x.index_fill_(1, torch.tensor([0], device=x.device), -1)
    return torch.where(x > 0, torch.full_like(x, 2), x)

def check(name, fn, backend, device, dynamic):
    x = torch.arange(1, 13, dtype=torch.int64, device=device).reshape(3, 4)
    xe, xc = x.clone(), x.clone()
    re = fn(xe)
    torch._dynamo.reset()
    rc = torch.compile(fn, backend=backend, dynamic=dynamic)(xc)
    ok = torch.equal(re, rc) and torch.equal(xe, xc)
    print(f"  {name:12s} [{backend}/{device} dynamic={dynamic}] {'OK' if ok else 'MISMATCH'}")
    if not ok:
        print("     eager   :", re.flatten().tolist(), " input after:", xe.flatten().tolist())
        print("     compiled:", rc.flatten().tolist(), " input after:", xc.flatten().tolist())

configs = [("aot_eager", "cpu")]
if torch.cuda.is_available():
    configs += [("aot_eager", "cuda"), ("inductor", "cuda")]
for backend, device in configs:
    for dyn in (None, True):
        for name, fn in [("min", f_min), ("fuzz", f_fuzz), ("slice", f_slice), ("masked_fill", f_masked_fill), ("where", f_where)]:
            check(name, fn, backend, device, dyn)
