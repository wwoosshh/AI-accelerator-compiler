import torch
print("torch", torch.__version__)
cases = {
 "contiguous()+unsqueeze_":  lambda x: (x.contiguous().unsqueeze_(2), x)[1],
 "contiguous()+transpose_":  lambda x: (x.contiguous().transpose_(0, 1), x)[1],
 "to(dtype)+unsqueeze_":     lambda x: (x.to(x.dtype).unsqueeze_(2), x)[1],
 "to(device)+unsqueeze_":    lambda x: (x.to(x.device).unsqueeze_(2), x)[1],
 "requires_grad_()+unsq_":   lambda x: (x.requires_grad_(False).unsqueeze_(2), x)[1],
 "detach()+unsqueeze_":      lambda x: (x.detach().unsqueeze_(2), x)[1],       # new object -> x unchanged
 "view_as()+unsqueeze_":     lambda x: (x.view_as(x).unsqueeze_(2), x)[1],     # new object -> x unchanged
 "direct unsqueeze_":        lambda x: (x.unsqueeze_(2), x)[1],
 "direct transpose_":        lambda x: (x.transpose_(0, 1), x)[1],
}
for name, f in cases.items():
    x = torch.arange(24, dtype=torch.int64).reshape(4, 6)
    xe = x.clone(); ref = f(xe)
    torch._dynamo.reset()
    xc = x.clone()
    try:
        out = torch.compile(f, backend="eager")(xc)
        ok = torch.equal(ref, out) and tuple(xe.shape) == tuple(xc.shape)
        print(f"[{name:26s}] {'OK' if ok else 'MISMATCH'}  eager x={tuple(xe.shape)} compiled x={tuple(xc.shape)}")
    except Exception as e:
        print(f"[{name:26s}] RAISES {type(e).__name__}: {str(e).splitlines()[0][:100]}")
