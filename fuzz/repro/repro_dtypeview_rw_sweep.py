import torch
print("torch", torch.__version__)
def f_full(t0):    v1 = t0.unsqueeze(2); v3 = t0.view(torch.int32) * 2; v3.sub_(-4); v1[:, 2:5, :] = 2; return v3.view(torch.int64)
def f_a(t0):       v3 = t0.view(torch.int32) * 2; v3.sub_(-4); t0[:, 2:5] = 2; return v3.view(torch.int64)   # no unsqueeze
def f_b(t0):       v1 = t0.unsqueeze(2); v3 = t0.view(torch.int32) * 2; v3.sub_(-4); v1[:, 2:5, :] = 2; return v3  # no view back
def f_c(t0):       v1 = t0.unsqueeze(2); v3 = t0.view(torch.int32) * 2; v1[:, 2:5, :] = 2; return v3.view(torch.int64)  # no sub_
def f_d(t0):       v1 = t0.unsqueeze(2); v3 = t0 * 2; v3.sub_(-4); v1[:, 2:5, :] = 2; return v3   # no dtype view
def f_e(t0):       v1 = t0.unsqueeze(2); v3 = t0.view(torch.int32) * 2; v3.add_(4); v1[:, 2:5, :] = 2; return v3.view(torch.int64)  # add_ instead of sub_
def f_f(t0):       v1 = t0.unsqueeze(2); v3 = t0.view(torch.int32) * 2; v3.sub_(-4); v1.fill_(2); return v3.view(torch.int64)  # fill_ via view
def f_g(t0):       v1 = t0.unsqueeze(2); v3 = t0.view(torch.int32) * 2; v3.sub_(-4); t0.fill_(2); return v3.view(torch.int64)  # fill_ direct
def run(name, f, backend, dynamic=None):
    x = torch.arange(-12, 12, dtype=torch.int64, device="cuda").reshape(4, 6)
    xe, xc = x.clone(), x.clone()
    re = f(xe); torch._dynamo.reset(); rc = torch.compile(f, backend=backend, dynamic=dynamic)(xc)
    ok = torch.equal(re, rc) and torch.equal(xe, xc)
    print(f"  {name:22s} [{backend:9s}] {'OK' if ok else 'MISMATCH'}")
for name, f in [("full", f_full), ("a no unsqueeze", f_a), ("b no view back", f_b), ("c no sub_", f_c), ("d no dtype view", f_d), ("e add_ not sub_", f_e), ("f fill_ via view", f_f), ("g fill_ direct", f_g)]:
    run(name, f, "aot_eager"); run(name, f, "inductor")
