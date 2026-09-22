import torch
def j_full(t2, src):     v1 = t2.reshape(t2.shape); v2 = v1.reshape((24,)); t2[:1, :8] = 5; v2.index_add_(0, torch.tensor([5], device=t2.device), src[:1]); return t2
def j_no_same(t2, src):  v2 = t2.reshape((24,)); t2[:1, :8] = 5; v2.index_add_(0, torch.tensor([5], device=t2.device), src[:1]); return t2
def j_no_set(t2, src):   v1 = t2.reshape(t2.shape); v2 = v1.reshape((24,)); v2.index_add_(0, torch.tensor([5], device=t2.device), src[:1]); return t2
def j_view(t2, src):     v1 = t2.view(t2.shape); v2 = v1.view((24,)); t2[:1, :8] = 5; v2.index_add_(0, torch.tensor([5], device=t2.device), src[:1]); return t2
def j_add_(t2, src):     v1 = t2.reshape(t2.shape); v2 = v1.reshape((24,)); t2[:1, :8] = 5; v2[5:6].add_(src[:1]); return t2
def j_setitem(t2, src):  v1 = t2.reshape(t2.shape); v2 = v1.reshape((24,)); t2[:1, :8] = 5; v2[5] = 7; return t2
def j_ret_v2(t2, src):   v1 = t2.reshape(t2.shape); v2 = v1.reshape((24,)); t2[:1, :8] = 5; v2.index_add_(0, torch.tensor([5], device=t2.device), src[:1]); return v2
def run(name, f, backend, dynamic):
    t2 = torch.arange(-12, 12, dtype=torch.int64, device="cuda").reshape(3, 8); src = torch.full((2,), 100, dtype=torch.int64, device="cuda")
    e2 = t2.clone(); er = f(e2, src.clone())
    torch._dynamo.reset(); c2 = t2.clone(); cr = torch.compile(f, backend=backend, dynamic=dynamic)(c2, src.clone())
    ok = torch.equal(er, cr) and torch.equal(e2, c2)
    print(f"  {name:10s} [{backend:9s} dyn={str(dynamic):5s}] {'OK' if ok else 'MISMATCH'}" + ("" if ok else f" eager t2[5]={e2.flatten()[5].item()} compiled t2[5]={c2.flatten()[5].item()}"))
print("torch", torch.__version__)
for name, f in [("full", j_full), ("no_same", j_no_same), ("no_set", j_no_set), ("view", j_view), ("add_", j_add_), ("setitem", j_setitem), ("ret_v2", j_ret_v2)]:
    for backend in ("aot_eager", "inductor"):
        for dyn in (None, True):
            run(name, f, backend, dyn)
