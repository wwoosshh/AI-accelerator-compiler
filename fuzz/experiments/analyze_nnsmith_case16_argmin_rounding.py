# -*- coding: utf-8 -*-
"""E1 (NNSmith 1차) 에서 같은 디바이스 재검증을 통과한 유일한 후보 #16 의 원인 확인.

그래프: tan -> interpolate(linear, size=58) -> argmin(dim=2). 입력 (1,38,1) 이라 보간된 각 행은 한 값의 복제여야 하는데,
eager CUDA 커널은 w0*a + w1*b 꼴로 계산해 행 안에서 최대 1.7 ulp 흔들리고, 컴파일 경로(분해)는 a + w*(b-a) 꼴로 정확히 상수 행을 만든다.
argmin 은 그 반올림 차이를 정수 인덱스 차이로 증폭한다(eager 51/36/53..., compiled 0). 값 자체는 0.87 ulp 이내로 같다 → 의미 버그가 아닌
부동소수점 반올림 차이. 사용: python analyze_nnsmith_case16_argmin_rounding.py <bug dir>
"""
import os, pickle, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
d = sys.argv[1]
x = torch.from_numpy(np.asarray(pickle.load(open(os.path.join(d, "oracle.pkl"), "rb"))["input"]["v6_0"])).cuda()
def interp(t): return torch.nn.functional.interpolate(torch.tan(t), size=[58], mode="linear")
with torch.no_grad():
    e = interp(x); spread = (e.max(2).values - e.min(2).values)[0]
    print("eager CUDA: rows not bit-identical %d/38, max spread %.2f ulp, argmin %s" % (
        int((spread != 0).sum()), (spread / e[0, :, 0].abs().clamp_min(1e-30)).max().item() / 2**-23, e.argmin(2)[0].tolist()[:12]))
    for backend in ("aot_eager", "inductor"):
        torch._dynamo.reset(); c = torch.compile(interp, backend=backend)(x)
        sp = (c.max(2).values - c.min(2).values)[0]
        print("%s: rows not bit-identical %d/38, |eager-compiled| max %.2f ulp, argmin %s" % (
            backend, int((sp != 0).sum()), ((e - c).abs() / e.abs().clamp_min(1e-30)).max().item() / 2**-23, c.argmin(2)[0].tolist()[:12]))
    ec = interp(x.cpu()); spc = (ec.max(2).values - ec.min(2).values)[0]
    print("eager CPU: rows not bit-identical %d/38 -> NNSmith 의 CPU-대-CUDA 오라클도 컴파일러와 무관하게 이 사례를 표시한다" % int((spc != 0).sum()))
