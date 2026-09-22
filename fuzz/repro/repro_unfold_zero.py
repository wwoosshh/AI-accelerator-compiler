# Bug candidate A: in-place mutation through a *gapped* unfold view (step > size)
# under torch.compile zeroes every base element that no window covers.
import sys, torch
print("torch", torch.__version__)

def f_ret(x):
    v = x.unfold(1, 1, 5)   # windows: columns 0 and 5 only (size=1, step=5)
    v.add_(v)               # doubles x[:, 0] and x[:, 5]; other columns must stay
    return x

def f_mut(x):               # same, but observed only through the mutated input
    v = x.unfold(1, 1, 5)
    v.add_(v)

def f_fill(x):              # variant: index_fill_ through a single-window unfold of a flat view
    v = x.view(24).unfold(0, 8, 24)   # shape (1, 8): covers elements 0..7 only
    v.index_fill_(1, torch.tensor([0, 4], device=x.device), -5)
    return x

def f_setitem(x):           # variant: setitem through an alias of an unfold view
    v = x.unfold(0, 1, 2)   # rows 0 and 2
    torch.ops.aten.alias(v)[0] = 7
    return x

def check(name, fn, backend, device, dynamic, via_input=False):
    x = torch.arange(1, 25, dtype=torch.int64, device=device).reshape(4, 6)
    xe, xc = x.clone(), x.clone()
    re = fn(xe)
    torch._dynamo.reset()
    rc = torch.compile(fn, backend=backend, dynamic=dynamic)(xc)
    if via_input:
        re, rc = xe, xc
    ok = torch.equal(re, rc)
    print(f"  {name:10s} [{backend}/{device} dynamic={dynamic}] {'OK' if ok else 'MISMATCH'}")
    if not ok:
        print("     eager   :", re.flatten().tolist())
        print("     compiled:", rc.flatten().tolist())

configs = [("aot_eager", "cpu")]
if torch.cuda.is_available() and "--cuda" in sys.argv:
    configs += [("inductor", "cuda"), ("aot_eager", "cuda")]
for backend, device in configs:
    for dyn in (None, True):
        check("ret", f_ret, backend, device, dyn)
        check("mut-input", f_mut, backend, device, dyn, via_input=True)
        check("index_fill", f_fill, backend, device, dyn)
        check("setitem", f_setitem, backend, device, dyn)
