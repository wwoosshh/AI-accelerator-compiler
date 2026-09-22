# Candidate I: a pointwise consumer of a dtype view of an input is evaluated AFTER a later in-place
# mutation of that input (read-after-write ordering violation) under Inductor.
import torch
print("torch", torch.__version__)
def i_dtype(x):        y = x.view(torch.int32) * 2; x[:, 2:5] = 2; return y.view(torch.int64)
def i_dtype_noback(x): y = x.view(torch.int32) * 2; x[:, 2:5] = 2; return y
def i_plain(x):        y = x * 2;                   x[:, 2:5] = 2; return y                 # control: no dtype view
def i_dtype_fill(x):   y = x.view(torch.int32) * 2; x.fill_(2);    return y                 # whole-tensor mutation
def i_dtype_add(x):    y = x.view(torch.int32) * 2; x.add_(1);     return y
def i_fuzz(t0):        v1 = t0.unsqueeze(2); v3 = t0.view(torch.int32) * 2; v3.sub_(-4); v1[:, 2:5, :] = 2; return v3.view(torch.int64)
def run(name, f, backend, dynamic=None):
    x = torch.arange(-12, 12, dtype=torch.int64, device="cuda").reshape(4, 6)
    xe, xc = x.clone(), x.clone()
    re = f(xe); torch._dynamo.reset(); rc = torch.compile(f, backend=backend, dynamic=dynamic)(xc)
    ok = torch.equal(re, rc) and torch.equal(xe, xc)
    print(f"  {name:16s} [{backend:9s} dyn={str(dynamic):4s}] {'OK' if ok else 'MISMATCH'}" + ("" if ok else f"  eager={re.flatten().tolist()[:6]} compiled={rc.flatten().tolist()[:6]}"))
for name, f in [("dtype+setitem", i_dtype), ("dtype, ret int32", i_dtype_noback), ("plain (control)", i_plain), ("dtype+fill_", i_dtype_fill), ("dtype+add_", i_dtype_add), ("fuzzer shape", i_fuzz)]:
    run(name, f, "aot_eager"); run(name, f, "inductor"); run(name, f, "inductor", True)
