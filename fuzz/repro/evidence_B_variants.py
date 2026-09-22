import torch
print("torch", torch.__version__)
def run(name, f):
    torch._dynamo.reset()
    cf = torch.compile(f, backend="aot_eager")
    base = torch.arange(24, dtype=torch.float64).reshape(4, 6)
    e0 = base.clone(); re = f(e0, e0[1:])
    c0 = base.clone(); rc = cf(c0, c0[1:])
    ok = torch.equal(re, rc) and torch.equal(e0, c0)
    print(f"[{name:28s}] {'OK' if ok else 'MISMATCH'}  eager={re.flatten().tolist()[:6]} compiled={rc.flatten().tolist()[:6]}"
          f"  offset eager={re.storage_offset()} compiled={rc.storage_offset()}")
def v1(t0, ta): t0.add_(100); return ta[0]          # found by fuzzer (minimal)
def v2(t0, ta): t0.add_(100); return ta             # return the aliased input itself
def v3(t0, ta): ta.add_(100); return ta[0]          # mutate the offset alias instead
def v4(t0, ta): ta.add_(100); return t0[1]          # mutate alias, return view of base
def v5(t0, ta): t0.add_(100); return ta.view(-1)    # flat view of alias
def v6(t0, ta): t0.add_(100); return t0[1]          # control: view of the mutated base itself
def v7(t0, ta): t0.add_(100); return ta[0] + 0      # control: non-alias output (copy)
for n, f in [("t0.add_; ret ta[0]", v1), ("t0.add_; ret ta", v2), ("ta.add_; ret ta[0]", v3), ("ta.add_; ret t0[1]", v4),
             ("t0.add_; ret ta.view(-1)", v5), ("ctrl t0.add_; ret t0[1]", v6), ("ctrl t0.add_; ret ta[0]+0", v7)]:
    run(n, f)
