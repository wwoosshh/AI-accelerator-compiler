# -*- coding: utf-8 -*-
"""aliasfuzz.py 를 wall-clock 예산 안에서 실행하는 드라이버.

eager 쪽 device-side assert 로 CUDA 컨텍스트가 오염되면 aliasfuzz.py 는 종료 코드 3 으로 끝난다
(같은 프로세스에서는 회복이 불가능). 이 드라이버는 그때 남은 예산으로 새 프로세스를 다음 시드로 다시 띄우고
(part_0, part_1, ...), 끝나면 부분 요약을 <out>/summary.json 하나로 합친다. 재시작 비용(torch import,
CUDA 초기화)은 예산 안에서 소모되므로 도구 자체의 비용으로 계산된다.

사용:
  python run_budget.py --minutes 20 --out results/E2b --seed 72 -- --backend inductor --device cuda --max-cases 1000
"""
import argparse
import json
import os
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--minutes', type=float, required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('rest', nargs=argparse.REMAINDER, help='-- 뒤의 인자는 aliasfuzz.py 로 그대로 전달')
a = ap.parse_args()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
extra = [x for x in a.rest if x != '--']
here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(a.out, exist_ok=True)
start = time.time()
deadline = start + a.minutes * 60
parts = []
part = 0
while True:
    remaining = deadline - time.time()
    if remaining < 30:
        break
    pdir = os.path.join(a.out, 'part_%d' % part)
    cmd = [sys.executable, os.path.join(here, 'aliasfuzz.py'), '--minutes', '%.3f' % (remaining / 60.0),
           '--seed', str(a.seed + part), '--out', pdir] + extra
    print('[run_budget] part %d: %s' % (part, ' '.join(cmd[1:])), flush=True)
    with open(os.path.join(a.out, 'part_%d.stdout' % part), 'w', encoding='utf-8', errors='replace') as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    summ = {}
    sp = os.path.join(pdir, 'summary.json')
    if os.path.exists(sp):
        summ = json.load(open(sp, encoding='utf-8'))
    parts.append({'part': part, 'dir': pdir, 'seed': a.seed + part, 'returncode': rc, 'summary': summ})
    print('[run_budget] part %d exited rc=%d aborted=%r iterations=%s stats=%s cases=%s' % (
        part, rc, summ.get('aborted'), summ.get('iterations'), summ.get('stats'), summ.get('n_cases')), flush=True)
    if rc != 3:
        break
    part += 1

agg = {'invalid': 0, 'compile_error': 0, 'value_diff': 0, 'alias_diff': 0, 'pass': 0}
iterations = n_cases = 0
fuzz_elapsed = 0.0
for p in parts:
    s = p['summary']
    for k in agg:
        agg[k] += s.get('stats', {}).get(k, 0)
    iterations += s.get('iterations', 0)
    n_cases += s.get('n_cases', 0)
    fuzz_elapsed += s.get('elapsed_s', 0.0)
result = {'budget_minutes': a.minutes, 'wall_elapsed_s': time.time() - start, 'fuzz_loop_elapsed_s': fuzz_elapsed,
          'restarts': len(parts) - 1, 'iterations': iterations, 'stats': agg, 'n_cases': n_cases,
          'parts': parts, 'torch': parts[0]['summary'].get('torch') if parts else None}
with open(os.path.join(a.out, 'summary.json'), 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=1)
print('[run_budget] DONE restarts=%d iterations=%d stats=%s cases=%d wall=%.0fs' % (
    result['restarts'], iterations, agg, n_cases, result['wall_elapsed_s']), flush=True)
