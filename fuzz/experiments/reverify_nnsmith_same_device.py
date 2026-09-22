# -*- coding: utf-8 -*-
"""Re-verify NNSmith INCONSISTENCY reports on the same device.

NNSmith's pt2 oracle compares CPU eager against CUDA torch.compile, so GPU-vs-CPU floating-point
differences are mixed into its "bugs". For every report directory we reload the model (gir.pkl +
model.pth) and the saved inputs, then compare
  (a) CUDA eager vs CUDA torch.compile(inductor)   -> mismatch = candidate compiler bug
  (b) CPU eager  vs CUDA eager                     -> mismatch = cross-device float difference
with NNSmith's own tolerance (rtol=1e-2, atol=1e-3).
Usage: python reverify_nnsmith_same_device.py <fuzz.root>
"""
import glob, os, pickle, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from nnsmith.materialize import Model
TorchModel = Model.init("torch", backend_target="cuda")

root = sys.argv[1]
dirs = sorted(d for d in glob.glob(os.path.join(root, "bug-*")) if "INCONSISTENCY" in d)
print(f"inconsistency reports: {len(dirs)}")
summary = {"same_device_mismatch": 0, "cross_device_only": 0, "not_reproduced": 0, "failed": 0}

def flat(o):
    if isinstance(o, dict): return list(o.values())
    if isinstance(o, (list, tuple)): return list(o)
    return [o]

def close(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    return torch.allclose(a, b, rtol=1e-2, atol=1e-3, equal_nan=True)

for d in dirs:
    name = os.path.basename(d)
    try:
        model = TorchModel.load(os.path.join(d, "model.pth"))
        oracle = pickle.load(open(os.path.join(d, "oracle.pkl"), "rb"))
        gir = pickle.load(open(os.path.join(d, "gir.pkl"), "rb"))
        ops = [type(inst.iexpr.op).__name__ for inst in gir.insts]
        inp = oracle["input"] if isinstance(oracle, dict) else oracle.input
        names = list(model.input_like.keys())
        args = [torch.from_numpy(np.asarray(inp[n])) for n in names]
        net = model.torch_model.eval()
        with torch.no_grad():
            out_cpu = net.cpu()(*args)
            net_cuda = net.cuda()
            cargs = [a.cuda() for a in args]
            out_cuda = net_cuda(*cargs)
            torch._dynamo.reset()
            out_comp = torch.compile(net_cuda, backend="inductor")(*cargs)
    except Exception as e:
        summary["failed"] += 1
        print(f"--- {name}: FAILED {type(e).__name__}: {str(e)[:140]}")
        continue
    same_ok = all(close(a, b) for a, b in zip(flat(out_cuda), flat(out_comp)))
    cross_ok = all(close(a, b) for a, b in zip(flat(out_cpu), flat(out_cuda)))
    dtypes = sorted(set(str(t.dtype) for t in flat(out_cuda)))
    if not same_ok:
        summary["same_device_mismatch"] += 1; tag = "SAME-DEVICE MISMATCH (candidate compiler bug)"
    elif not cross_ok:
        summary["cross_device_only"] += 1; tag = "same device OK; cross-device (CPU vs CUDA eager) float difference only"
    else:
        summary["not_reproduced"] += 1; tag = "same device OK; original report not reproduced"
    print(f"--- {name}: {tag}\n    ops={ops}\n    out dtypes={dtypes}")
print("summary:", summary)
