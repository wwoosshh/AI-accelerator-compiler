# -*- coding: utf-8 -*-
"""사례 자동 분류: unfold 계열(A) / 별칭 입력 계열(B) / dtype 뷰 별칭(C) / 기타"""
import glob, json, re, sys, collections
dirs = sys.argv[1:] or sorted(glob.glob('results/*/'))
def classify(c):
    src = c['source']
    kinds = set(c['kinds'])
    if 'view_unfold' in kinds and re.search(r'unfold\(', src):
        return 'A_unfold'
    if c.get('aliased'):
        return 'B_aliased_input'
    if c['status'] == 'alias_diff' and ('noop_arith' in kinds or re.search(r'(\+ 0|\* 1|[-+] k|\* k)\s*$', src, re.M)):
        return 'K_known_197893'   # k 가 0 또는 1 이면 x-0 / x+0 / x*1 과 같음
    if c['status'] == 'alias_diff' and 'view_dtype' in kinds:
        return 'C_dtype_view_alias'
    if 'index_inplace' in kinds and re.search(r'\[\w+ > -?\d+\] = ', src) and c['status'] == 'value_diff':
        return 'D_inductor_indexfill_mask'
    return 'other'
for d in dirs:
    files = sorted(glob.glob(d + '/case_*.json'))
    if not files:
        continue
    cnt = collections.Counter()
    others = []
    for f in files:
        c = json.load(open(f, encoding='utf-8'))
        cls = classify(c)
        cnt[cls] += 1
        if cls == 'other':
            others.append((f, c))
    print('==', d, dict(cnt))
    for f, c in others:
        print('  --- OTHER:', f, c['status'], 'dynamic=', c['dynamic'], 'aliased=', c.get('aliased'))
        print('     ' + c['source'].replace('\n', '\n     ').rstrip())
        for x in c['diffs'][:2]:
            print('      >', x[:200])
