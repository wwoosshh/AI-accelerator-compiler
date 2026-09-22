import sys, torch
print("torch", torch.__version__)
def idx(x, i): return torch.tensor([i], device=x.device)
def a_slice_add(x):  x[:, :] += 5; x.index_fill_(1, idx(x, 2), -1); x[x > 0] = 2; return x     # found
def b_add(x):        x += 5;       x.index_fill_(1, idx(x, 2), -1); x[x > 0] = 2; return x     # no slice
def c_add_(x):       x.add_(5);    x.index_fill_(1, idx(x, 2), -1); x[x > 0] = 2; return x
def d_neg(x):        x.neg_();     x.index_fill_(1, idx(x, 2), -1); x[x > -100] = 2; return x  # fuzzer case E shape
def e_col_assign(x): x.add_(5);    x[:, 2] = -1;                    x[x > 0] = 2; return x     # slice instead of index_fill_
def f_local(x0):     x = x0.clone(); x.add_(5); x.index_fill_(1, idx(x, 2), -1); x[x > 0] = 2; return x  # not an input
def g_noreturn(x):   x.add_(5);    x.index_fill_(1, idx(x, 2), -1); x[x > 0] = 2                # mutation only
def h_masked_fill(x):x.add_(5);    x.index_fill_(1, idx(x, 2), -1); x.masked_fill_(x > 0, 2); return x
def i_where(x):      x.add_(5);    x.index_fill_(1, idx(x, 2), -1); return torch.where(x > 0, torch.full_like(x, 2), x)
def j_two_only(x):   x.index_fill_(1, idx(x, 2), -1); x[x > 0] = 2; return x                    # control (passes)
def check(name, fn, backend):
    x = torch.arange(1, 13, dtype=torch.int64, device="cuda").reshape(3, 4)
    xe, xc = x.clone(), x.clone()
    re = fn(xe); torch._dynamo.reset(); rc = torch.compile(fn, backend=backend)(xc)
    ok = (re is None or torch.equal(re, rc)) and torch.equal(xe, xc)
    print(f"  {name:14s} [{backend}] {'OK' if ok else 'MISMATCH'}" + ("" if ok else f"  eager={xe.flatten().tolist()} compiled={xc.flatten().tolist()}"))
for name, fn in [("a_slice_add", a_slice_add), ("b_add", b_add), ("c_add_", c_add_), ("d_neg", d_neg), ("e_col_assign", e_col_assign),
                 ("f_local", f_local), ("g_noreturn", g_noreturn), ("h_masked_fill", h_masked_fill), ("i_where", i_where), ("j_two_only", j_two_only)]:
    for backend in ("aot_eager", "inductor"):
        check(name, fn, backend)
