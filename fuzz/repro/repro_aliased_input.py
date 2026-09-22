# Bug candidate B: two inputs that alias each other (ta = t0[1:]); mutating t0 in place and
# returning a strided view of ta gives wrong values under torch.compile, and different wrong
# values on the second call.
import sys, torch
print("torch", torch.__version__)

def fn(t0, ta):
    t0.clamp_(0, 3)
    return ta[2::2, 2::3, :]

def run(backend, device, dynamic):
    torch._dynamo.reset()
    cf = torch.compile(fn, backend=backend, dynamic=dynamic)
    for call in range(3):
        g = torch.Generator().manual_seed(call)
        base = torch.randint(-8, 9, (4, 3, 2), generator=g, dtype=torch.float64).to(device)
        e0 = base.clone(); re = fn(e0, e0[1:])
        c0 = base.clone(); rc = cf(c0, c0[1:])
        ok_out, ok_in = torch.equal(re, rc), torch.equal(e0, c0)
        print(f"  [{backend}/{device} dynamic={dynamic}] call{call}: out {'OK' if ok_out else 'MISMATCH'}, "
              f"mutated input {'OK' if ok_in else 'MISMATCH'}")
        if not ok_out:
            print("     expected (eager) :", re.flatten().tolist(), " = t0[3, 2, :] after clamp =", e0[3, 2, :].tolist())
            print("     got (compiled)   :", rc.flatten().tolist())
        if not ok_in:
            print("     t0 eager   :", e0.flatten().tolist())
            print("     t0 compiled:", c0.flatten().tolist())

configs = [("aot_eager", "cpu")]
if torch.cuda.is_available() and "--cuda" in sys.argv:
    configs += [("inductor", "cuda")]
for backend, device in configs:
    for dyn in (None, True):
        run(backend, device, dyn)
