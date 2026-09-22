# -*- coding: utf-8 -*-
"""triage.py 가 'other' 로 분류한 사례의 소스와 diff 를 출력. 사용: python experiments/show_other_cases.py <results dir>..."""
import glob, json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "..", "triage.py"), encoding="utf-8").read().split("for d in dirs:")[0]
ns = {}; exec(src, ns); classify = ns["classify"]
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "case_*.json"))):
        c = json.load(open(f, encoding="utf-8"))
        if classify(c) != "other":
            continue
        print("=====", f, c["status"], "dynamic=", c.get("dynamic"), "aliased=", c.get("aliased"))
        print(c["source"].rstrip())
        for x in c["diffs"][:4]:
            print("   ", x[:200])
        m = f.replace("case_", "min_case_").replace(".json", ".py")
        if os.path.exists(m):
            body = open(m, encoding="utf-8").read()
            i = body.find("def fn("); j = body.find("\n\n", i)
            print("--- minimized:"); print(body[i:j] if i >= 0 else "(n/a)")
