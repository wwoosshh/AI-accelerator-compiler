# -*- coding: utf-8 -*-
"""E3: which oracle detected each divergence, per bug family.

Usage (from fuzz/): python experiments/e3_oracle_ablation.py results/smoke_aot results/smoke_ind results/A_aot_cpu results/B_ind_cuda
Diff path conventions written by aliasfuzz.py:
  callN.out[...]      -> return-value oracle
  callN.argK          -> mutated-input-state oracle (only if no .out diff)
  probe.*             -> alias-probe oracle (status alias_diff)
  everything on call1 -> second-call (guard/cache) oracle only
"""
import collections
import glob
import json
import os
import re
import sys

here = os.path.dirname(os.path.abspath(__file__))
triage_src = open(os.path.join(here, "..", "triage.py"), encoding="utf-8").read().split("for d in dirs:")[0]
ns = {}
exec(triage_src, ns)
classify = ns["classify"]

runs = sys.argv[1:] or ["results/smoke_aot", "results/smoke_ind", "results/A_aot_cpu", "results/B_ind_cuda"]
rows = []
for d in runs:
    for f in sorted(glob.glob(os.path.join(d, "case_*.json"))):
        c = json.load(open(f, encoding="utf-8"))
        diffs = c["diffs"]
        fam = classify(c)
        s = c["source"]
        if fam == "other":
            if re.search(r"_scatter\((\w+), \1[.\[]", s):
                fam = "H1"
            elif "_scatter(" in s:
                fam = "H2"
            elif re.search(r"\.view\(torch\.int", s) and c["status"] == "value_diff":
                fam = "I"
            elif re.search(r"index_(fill|add|copy)_", s) and ".clone()" in s:
                fam = "H3"
        value_out = any(x.startswith("call") and ".out" in x for x in diffs)
        value_arg_only = (not value_out) and any(x.startswith("call") and ".arg" in x for x in diffs)
        probe_only = c["status"] == "alias_diff"
        call1_only = all(x.startswith("call1") or x.startswith("probe.call1") for x in diffs)
        rows.append((os.path.basename(d.rstrip("/\\")), fam, c["status"], value_out, value_arg_only, probe_only, call1_only))

fams = sorted(set(r[1] for r in rows))
print(f"{'family':28s} {'cases':>5s} {'value(out)':>10s} {'input-state only':>17s} {'alias-probe only':>17s} {'2nd-call only':>13s}")
for fam in fams:
    rs = [r for r in rows if r[1] == fam]
    print(f"{fam:28s} {len(rs):5d} {sum(r[3] for r in rs):10d} {sum(r[4] for r in rs):17d} {sum(r[5] for r in rs):17d} {sum(r[6] for r in rs):13d}")
print(f"{'TOTAL':28s} {len(rows):5d} {sum(r[3] for r in rows):10d} {sum(r[4] for r in rows):17d} {sum(r[5] for r in rows):17d} {sum(r[6] for r in rows):13d}")
