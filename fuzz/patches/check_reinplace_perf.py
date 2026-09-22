import torch, re, io, contextlib, logging
torch._logging.set_logs(output_code=True)
def gen_code(f, *args):
    torch._dynamo.reset()
    buf = io.StringIO()
    h = logging.StreamHandler(buf); log = logging.getLogger("torch._inductor.codecache"); log.addHandler(h)
    torch.compile(f, backend="inductor")(*args)
    log.removeHandler(h)
    code = buf.getvalue()
    call = code[code.find("def call(self, args):"):]
    call = call[:call.find("def benchmark_compiled_module")] if "def benchmark_compiled_module" in call else call
    return call
x = torch.arange(24, dtype=torch.int64, device="cuda").reshape(4, 6); src = torch.ones(2, 6, dtype=torch.int64, device="cuda")
def legit1(x, src): x[1:3] = src; return x                                 # in-place slice assignment (should reinplace)
def legit2(x, src): y = torch.slice_scatter(x, src, 0, 1, 3); x.copy_(y); return x   # result written back (should reinplace)
def bug_h2(x, src): y = torch.slice_scatter(x, src, 0, 1, 3); x.zero_(); return y    # must NOT reinplace
for name, f in [("legit1 x[1:3]=src", legit1), ("legit2 copy_(x,y)", legit2), ("bug_h2", bug_h2)]:
    call = gen_code(f, x.clone(), src.clone())
    allocs = re.findall(r"empty_strided_cuda\(\((\d+), (\d+)\)", call)
    print(f"{name:22s} full-size (4,6) allocations in call(): {sum(1 for a in allocs if a == ('4','6'))}  | all allocs: {allocs}")
