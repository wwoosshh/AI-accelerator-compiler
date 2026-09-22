import torch
print("torch", torch.__version__)
def f_direct(x):  x.unsqueeze_(0); return x
def f_squeeze(x): x.squeeze_(0); return x            # x has shape (1, 4, 6)
def f_contig(x):  v = x.contiguous(); v.unsqueeze_(2); return x
def f_t_(x):      x.t_(); return x                   # rank-preserving control
for name, f, shape in [("x.unsqueeze_(0)", f_direct, (4, 6)), ("x.squeeze_(0)", f_squeeze, (1, 4, 6)), ("contiguous()+unsqueeze_", f_contig, (4, 6)), ("x.t_() control", f_t_, (4, 6))]:
    x = torch.arange(24, dtype=torch.int64).reshape(shape)
    ref = f(x.clone())
    torch._dynamo.reset()
    try:
        out = torch.compile(f, backend="eager")(x.clone())
        print(f"[{name:26s}] OK={torch.equal(ref, out)} eager={tuple(ref.shape)} compiled={tuple(out.shape)}")
    except Exception as e:
        print(f"[{name:26s}] RAISES {type(e).__name__}: {str(e).splitlines()[0][:110]}")
