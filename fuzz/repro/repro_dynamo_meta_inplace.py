# Candidate F (crash, not silent): in-place metadata mutation (unsqueeze_/transpose_) on a no-op alias of a
# graph input makes Dynamo fail while building guards ("Guard failed on the same frame it was created.
# This is a bug" / IndexError in build_guards). Eager works.
import torch, traceback
print("torch", torch.__version__)
def f1(x):
    v = x.contiguous()          # x is contiguous -> v is x (no-op alias)
    v.unsqueeze_(2)
    return x
def f2(x):
    v = x.contiguous(); w = v.view(torch.int32)
    v.unsqueeze_(2); w.clamp_(-3, 4)
    return w, x
def f3(x):
    x.transpose_(0, 1)          # metadata mutation directly on the input
    return x
def f4(x):
    v = x.contiguous(); v.transpose_(0, 1)
    return x
def f5(x):
    v = x.view_as(x); v.unsqueeze_(0)
    return v
for name, f in [("contig+unsqueeze_", f1), ("contig+dtypeview+unsqueeze_", f2), ("input.transpose_", f3), ("contig+transpose_", f4), ("view_as+unsqueeze_", f5)]:
    for backend in ("eager", "aot_eager"):
        x = torch.arange(24, dtype=torch.int64).reshape(4, 6)
        try:
            ref = f(x.clone())
        except Exception as e:
            print(f"[{name:30s}] eager itself raises {type(e).__name__}: {str(e)[:80]}"); break
        torch._dynamo.reset()
        try:
            out = torch.compile(f, backend=backend)(x.clone())
            def flat(o): return [t.shape for t in (o if isinstance(o, tuple) else (o,))]
            same = all(torch.equal(a, b) for a, b in zip(ref if isinstance(ref, tuple) else (ref,), out if isinstance(out, tuple) else (out,)))
            print(f"[{name:30s}] {backend:9s} OK={same} shapes eager={flat(ref)} compiled={flat(out)}")
        except Exception as e:
            msg = str(e).replace("\n", " ")[:150]
            print(f"[{name:30s}] {backend:9s} RAISES {type(e).__name__}: {msg}")
