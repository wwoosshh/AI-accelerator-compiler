# -*- coding: utf-8 -*-
"""NNSmith INCONSISTENCY 보고를 NNSmith 자신의 컴파일 경로로 재검증하고 불일치를 분류한다.

v1 은 `torch.compile(net)` 로 다시 컴파일했는데, NNSmith 의 SymbolNet.forward 는 해석기 루프라 Dynamo 가 곳곳에서 그래프를
끊어 사실상 eager 로 실행됐고(그래서 '재현 안 됨' 이 대부분이었다). NNSmith 의 pt2 백엔드는 `torch.fx.symbolic_trace` 로 먼저
그래프를 뽑고 `torch.compile(traced, fullgraph=True)` 한다. v2 는 그 경로(`factory.make_backend`)를 그대로 쓴다.
참조값(oracle.output)은 생성 시 CUDA eager 로 계산된 것이라 같은 디바이스 비교다(CPU 가 아님).

분류(사례별, 불일치 출력들을 보고):
  int          정수/불 출력만 다름 (argmax/argmin 동률, 반올림 경계 → 정수 증폭)
  lowprec      다른 출력의 최대 절대 오차가 출력 스케일(max|eager|)의 32 ulp 이하 (fp16 3.1%, fp32 3.8e-6, fp64 7e-15).
               Inductor 는 fp16 연쇄를 fp32 로 계산해 한 번만 반올림하므로 eager 와 반올림 순서가 다르다
  overflow     fp16/bf16 에서 한쪽만 inf/nan (fp32 내부 계산은 넘치지 않음)
  discontinuous 위 기준을 넘지만 그래프에 불연속 연산(Ceil/Floor/Round/ArgMax/ArgMin/비교/Where/정수 캐스트)이 있어
               미세한 반올림 차이가 큰 출력 차이로 증폭될 수 있는 경우
  LARGE        위 어디에도 해당하지 않는 큰 차이 → 수작업 확인 대상(진짜 의미 버그 후보)
  trace        symbolic_trace 된 모듈을 컴파일 없이 실행해도 이미 다름 → 컴파일러가 아닌 추적(하네스) 문제
사용: python reverify_nnsmith_same_device.py <fuzz.root>
"""
import glob, os, pickle, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from nnsmith.materialize import Model, TestCase
from nnsmith.backends import BackendFactory
ModelType = Model.init("torch", backend_target="cuda")
factory = BackendFactory.init("pt2", target="cuda", optmax=True)
RTOL, ATOL = 1e-2, 1e-3
root = sys.argv[1]
dirs = sorted(d for d in glob.glob(os.path.join(root, "bug-*")) if "INCONSISTENCY" in d)
print(f"inconsistency reports: {len(dirs)} | compile path: fx.symbolic_trace + torch.compile(fullgraph=True, backend=inductor, mode={factory.mode})")

def T(x): return torch.as_tensor(np.asarray(x))
def close(a, b): return torch.allclose(T(a).float(), T(b).float(), rtol=RTOL, atol=ATOL, equal_nan=True)

DISCONT = {"ArgMax", "ArgMin", "Ceil", "Floor", "Round", "Trunc", "CastBool", "CastI32", "CastI64", "CastU8",
           "CastI8", "Greater", "Less", "Equal", "Where", "Sign", "GreaterEqual", "LessEqual", "NotEqual"}

def describe(e, c):
    e, c = T(e), T(c); dt = c.dtype
    ef, cf = e.float(), c.float()
    bad = ~torch.isclose(ef, cf, rtol=RTOL, atol=ATOL, equal_nan=True)
    n = int(bad.sum())
    scale = ef.nan_to_num(0).abs().max().item()
    maxabs = (ef - cf).abs().nan_to_num(0)[bad].max().item()
    if not dt.is_floating_point:
        return {"dtype": str(dt), "n": n, "kind": "int", "maxabs": maxabs, "scale": scale}
    nonfinite = (torch.isfinite(ef) != torch.isfinite(cf))[bad].any().item()
    eps = torch.finfo(dt).eps
    ulps = maxabs / max(scale, 1e-30) / eps
    if nonfinite:
        kind = "overflow" if dt in (torch.float16, torch.bfloat16) else "LARGE"
    elif ulps <= 32:
        kind = "lowprec"
    else:
        kind = "LARGE"
    return {"dtype": str(dt), "n": n, "kind": kind, "maxabs": maxabs, "scale": scale, "ulps": round(ulps, 1)}

summary = {"int": 0, "lowprec": 0, "overflow": 0, "discontinuous": 0, "LARGE": 0, "trace": 0, "not_reproduced": 0, "failed": 0}
for d in dirs:
    name = os.path.basename(d)
    try:
        tc = TestCase.load(model_type=ModelType, root_folder=d)
        model, inputs = tc.model, tc.oracle.input
        gir = pickle.load(open(os.path.join(d, "gir.pkl"), "rb"))
        ops = [type(i.iexpr.op).__name__ for i in gir.insts if type(i.iexpr.op).__name__ != "Input"]
        net = model.torch_model.cuda().eval()
        names, onames = list(model.input_like.keys()), list(model.output_like.keys())
        args = [torch.from_numpy(np.asarray(inputs[n])).cuda() for n in names]
        with torch.no_grad():
            eager = {k: v.cpu().numpy() for k, v in zip(onames, net(*args))}
        comp = factory.make_backend(model)(inputs)
        # traced module run eagerly (harness check) - use NNSmith's tracing context
        from nnsmith.materialize.torch.symbolnet import FxTracing
        with torch.no_grad(), FxTracing():
            tr = torch.fx.symbolic_trace(net)
        with torch.no_grad():
            tr_out = {k: v.cpu().numpy() for k, v in zip(onames, tr(*args))}
    except Exception as e:
        summary["failed"] += 1; print(f"--- {name}: FAILED {type(e).__name__}: {str(e)[:160]}"); continue
    bad = [k for k in onames if not close(eager[k], comp[k])]
    if not bad:
        summary["not_reproduced"] += 1; print(f"--- {name}: not reproduced (NNSmith path == eager) ops={ops}"); continue
    trace_bad = [k for k in onames if not close(eager[k], tr_out[k])]
    if trace_bad:
        summary["trace"] += 1; print(f"--- {name}: TRACE artifact (traced module differs from eager without compile) outs={trace_bad} ops={ops}"); continue
    descs = {k: describe(eager[k], comp[k]) for k in bad}
    kinds = {v["kind"] for v in descs.values()}
    verdict = "LARGE" if "LARGE" in kinds else ("overflow" if "overflow" in kinds else ("lowprec" if "lowprec" in kinds else "int"))
    if verdict == "LARGE" and DISCONT & set(ops):
        verdict = "discontinuous"
    summary[verdict] += 1
    print(f"--- {name}: {verdict} | " + "; ".join(f"{k}:{v['dtype']} n={v['n']} maxabs={v['maxabs']:.4g} scale={v['scale']:.4g}" + (f" ulps={v['ulps']}" if 'ulps' in v else "") for k, v in descs.items()) + f" | ops={ops}")
print("summary:", summary)
