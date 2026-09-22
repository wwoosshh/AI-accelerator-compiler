# -*- coding: utf-8 -*-
"""NNSmith fuzz.root 디렉터리를 분석: 증상별 개수, 불일치 사례의 연산 목록과 최대 오차, 정확 비교 재검증.

사용: python analyze_nnsmith_report.py <fuzz.root> [<log file>]
"""
import glob
import json
import os
import pickle
import re
import sys
import warnings

warnings.filterwarnings("ignore")
import torch  # noqa: E402

root = sys.argv[1]
log = sys.argv[2] if len(sys.argv) > 2 else None
dirs = sorted(glob.glob(os.path.join(root, "bug-*")))
print(f"report dirs: {len(dirs)}")
by_sym = {}
for d in dirs:
    m = re.match(r"bug-Symptom\.(\w+)-Stage\.(\w+)-(\d+)", os.path.basename(d))
    key = f"{m.group(1)}/{m.group(2)}" if m else "?"
    by_sym.setdefault(key, []).append(d)
for k, v in by_sym.items():
    print(f"  {k}: {len(v)}")

if log and os.path.exists(log):
    txt = open(log, encoding="utf-8", errors="replace").read()
    for pat in (r"Total (\d+) testcases generated", r"Total (\d+) bugs found", r"Total (\d+) failed to make testcases"):
        mm = re.findall(pat, txt)
        if mm:
            print(f"  log: {pat.split('(')[0].strip()} -> {mm[-1]}")

print("\n== inconsistency cases ==")
for d in by_sym.get("INCONSISTENCY/VERIFICATION", []):
    print(f"--- {os.path.basename(d)}")
    try:
        gir = pickle.load(open(os.path.join(d, "gir.pkl"), "rb"))
        ops = []
        for inst in gir.insts:
            ops.append(type(inst.iexpr.op).__name__)
        print("   ops:", ops)
    except Exception as e:
        print("   gir load failed:", type(e).__name__, e)
    err = os.path.join(d, "err.log")
    if os.path.exists(err):
        lines = [l.strip() for l in open(err, encoding="utf-8", errors="replace") if l.strip()]
        for l in lines[:3]:
            print("   err:", l[:200])
    try:
        oracle = pickle.load(open(os.path.join(d, "oracle.pkl"), "rb"))
        inp, out = (oracle["input"], oracle["output"]) if isinstance(oracle, dict) else (oracle.input, oracle.output)
        dts = {k: str(v.dtype) if hasattr(v, "dtype") else type(v).__name__ for k, v in {**inp, **out}.items()}
        print("   dtypes:", dts)
    except Exception as e:
        print("   oracle load failed:", type(e).__name__, e)
