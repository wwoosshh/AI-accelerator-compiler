# -*- coding: utf-8 -*-
"""NNSmith 불일치가 '컴파일러 버그' 인지 '수치적으로 불안정한 그래프' 인지 가르는 조건수 검사.

각 보고에 대해 (1) eager 와 NNSmith 컴파일 경로(symbolic_trace + compile fullgraph)의 출력 차이, (2) float 입력을 1 ulp 만큼
무작위 부호로 흔든 뒤 eager 만 다시 실행했을 때의 출력 변화를 잰다. (2) 가 (1) 과 같은 크기이거나 더 크면, 그 그래프는
입력의 마지막 비트에도 그만큼 흔들리는 불안정한 함수라 eager 대 compiled 차이를 버그로 볼 수 없다(반올림 순서 차이면 충분).
사용: python nnsmith_condition_test.py <fuzz.root>
"""
import glob, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from nnsmith.materialize import Model, TestCase
from nnsmith.backends import BackendFactory
ModelType = Model.init("torch", backend_target="cuda")
factory = BackendFactory.init("pt2", target="cuda", optmax=True)
RTOL, ATOL = 1e-2, 1e-3
root = sys.argv[1]
dirs = sorted(d for d in glob.glob(os.path.join(root, "bug-*")) if "INCONSISTENCY" in d)

def T(x): return torch.as_tensor(np.asarray(x)).float()
def close(a, b): return torch.allclose(T(a), T(b), rtol=RTOL, atol=ATOL, equal_nan=True)
def maxabs(a, b): return (T(a) - T(b)).abs().nan_to_num(0).max().item()

def perturb(arr, g):
    t = torch.as_tensor(np.asarray(arr))
    if not t.is_floating_point():
        return arr
    sign = torch.where(torch.rand(t.shape, generator=g) < 0.5, -1.0, 1.0).to(t.dtype)
    eps = torch.finfo(t.dtype).eps
    return (t + sign * t.abs().clamp_min(torch.finfo(t.dtype).tiny) * eps).numpy()

summary = {"unstable_graph": 0, "stable_but_differs": 0, "not_reproduced": 0, "failed": 0}
g = torch.Generator().manual_seed(0)
for d in dirs:
    name = os.path.basename(d).replace("bug-Symptom.INCONSISTENCY-Stage.VERIFICATION-", "#")
    try:
        tc = TestCase.load(model_type=ModelType, root_folder=d)
        model, inputs = tc.model, tc.oracle.input
        net = model.torch_model.cuda().eval()
        names, onames = list(model.input_like.keys()), list(model.output_like.keys())
        def run_eager(inp):
            with torch.no_grad():
                return {k: v.cpu().numpy() for k, v in zip(onames, net(*[torch.from_numpy(np.asarray(inp[n])).cuda() for n in names]))}
        eager = run_eager(inputs)
        comp = factory.make_backend(model)(inputs)
        pert = {n: perturb(inputs[n], g) for n in names}
        eager_p = run_eager(pert)
    except Exception as e:
        summary["failed"] += 1; print(f"--- {name}: FAILED {type(e).__name__}: {str(e)[:120]}"); continue
    bad = [k for k in onames if not close(eager[k], comp[k])]
    if not bad:
        summary["not_reproduced"] += 1; print(f"--- {name}: not reproduced"); continue
    rows = []
    unstable = True
    for k in bad:
        dc, dp = maxabs(eager[k], comp[k]), maxabs(eager[k], eager_p[k])
        tol_fail_p = not close(eager[k], eager_p[k])
        rows.append(f"{k}: compiled diff {dc:.4g} | 1-ulp input perturbation diff {dp:.4g}{' (fails NNSmith tol)' if tol_fail_p else ''}")
        if not (dp >= 0.5 * dc or tol_fail_p):
            unstable = False
    verdict = "unstable_graph" if unstable else "stable_but_differs"
    summary[verdict] += 1
    print(f"--- {name}: {verdict} | " + "; ".join(rows))
print("summary:", summary)
