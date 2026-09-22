# Three Inductor-only families seen in run B (functional results corrupted by a LATER in-place mutation of an input).
import torch
print("torch", torch.__version__)
def idx(x, *v): return torch.tensor(list(v), device=x.device)
# H1: no-op *_scatter(x, x.view) returns an alias of x
def h1_diag(t1):   v2 = torch.diagonal_scatter(t1, t1.diagonal()); t1.fill_(2); return v2
def h1_select(t0): v2 = torch.select_scatter(t0, t0.select(0, 1), 0, 1); t0.fill_(2); return v2
def h1_slice(t0):  v2 = torch.slice_scatter(t0, t0[1:3], dim=0, start=1, end=3); t0.fill_(2); return v2
# H2: scatter result reinplaced into an input buffer that is mutated afterwards
def h2_slice(t0, t1): v1 = torch.slice_scatter(t1, t0[1:3], dim=0, start=1, end=3); t1.zero_(); return v1
def h2_select(t0, t1): v1 = torch.select_scatter(t1, t0.select(0, 2), 0, 2); t1.fill_(7); return v1
def h2_copy(t0, t1):   v1 = t1.clone(); v1[:, :] = t0[:, :]; t0.fill_(9); return v1
# H3: clone of an input removed although the input is mutated afterwards
def h3_plain(t0):     v2 = t0.clone(); t0.clamp_(-4, 4); return v2
def h3_after_fill(t0):t0.index_fill_(1, idx(t0, 1), 4); v2 = t0.clone(); t0.clamp_(-4, 4); return v2
def h3_after_add(t0): t0.add_(1); v2 = t0.clone(); t0.clamp_(-4, 4); return v2

def run(name, f, nargs, backend, dynamic=None):
    base = [torch.arange(-12, 12, dtype=torch.int64, device="cuda").reshape(4, 6), torch.arange(24, dtype=torch.int64, device="cuda").reshape(4, 6)][:nargs]
    ea = [b.clone() for b in base]; re = f(*ea)
    torch._dynamo.reset()
    ca = [b.clone() for b in base]; rc = torch.compile(f, backend=backend, dynamic=dynamic)(*ca)
    ok = torch.equal(re, rc) and all(torch.equal(a, b) for a, b in zip(ea, ca))
    print(f"  {name:14s} [{backend:9s} dyn={str(dynamic):4s}] {'OK' if ok else 'MISMATCH'}" + ("" if ok else f"  eager={re.flatten().tolist()[:8]} compiled={rc.flatten().tolist()[:8]}"))
for name, f, n in [("H1 diag", h1_diag, 1), ("H1 select", h1_select, 1), ("H1 slice", h1_slice, 1),
                   ("H2 slice", h2_slice, 2), ("H2 select", h2_select, 2), ("H2 copy", h2_copy, 2),
                   ("H3 plain", h3_plain, 1), ("H3 after_fill", h3_after_fill, 1), ("H3 after_add", h3_after_add, 1)]:
    run(name, f, n, "aot_eager")
    run(name, f, n, "inductor")
    run(name, f, n, "inductor", True)
