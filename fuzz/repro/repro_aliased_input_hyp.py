# Hypothesis: the returned view is regenerated from the synthetic base t0 instead of from ta = t0[1:],
# i.e. compiled returns t0[2::2, 2::3, :] (= t0[2, 2, :]) instead of ta[2::2, 2::3, :] (= t0[3, 2, :]).
import torch
print("torch", torch.__version__)

def fn(t0, ta):
    t0.clamp_(0, 3)
    return ta[2::2, 2::3, :]

def fn_simple(t0, ta):      # even simpler: return the aliased input's slice without stride tricks
    t0.add_(100)
    return ta[0]

def fn_noret_alias(t0, ta):  # control: no mutation -> should be fine
    return ta[2::2, 2::3, :]

for name, f in [("clamp+strided", fn), ("add_+ta[0]", fn_simple), ("no-mutation", fn_noret_alias)]:
    torch._dynamo.reset()
    cf = torch.compile(f, backend="aot_eager")
    base = torch.arange(24, dtype=torch.float64).reshape(4, 3, 2)
    e0 = base.clone(); re = f(e0, e0[1:])
    c0 = base.clone(); rc = cf(c0, c0[1:])
    print(f"[{name}] eager out={re.flatten().tolist()}  compiled out={rc.flatten().tolist()}  "
          f"{'OK' if torch.equal(re, rc) else 'MISMATCH'}")
    if name == "clamp+strided":
        print("   t0[3,2,:] (correct) =", c0[3, 2, :].tolist(), "  t0[2,2,:] (offset-dropped hypothesis) =", c0[2, 2, :].tolist())
    if name == "add_+ta[0]":
        print("   ta[0] = t0[1] (correct) =", c0[1].flatten().tolist(), "  t0[0] (offset-dropped) =", c0[0].flatten().tolist())
    # does the returned view alias the input storage at all?
    print("   compiled out shares storage with t0:", rc.untyped_storage().data_ptr() == c0.untyped_storage().data_ptr(),
          " storage_offset:", rc.storage_offset(), " expected offset:", re.storage_offset())
