# -*- coding: utf-8 -*-
# aliasfuzz repro | status=value_diff backend=inductor device=cuda seed=71001416
import torch

def fn(t0, t1, k, lst, dct):
    v1 = t0[:, :]
    v2 = t0.unfold(0, 3, 3)
    lst.append(t1.shape[1] + k)
    v2.clamp_(-4, 3)
    v3 = t0.transpose(0, 1)
    v1[:-1, :] = v1[1:, :].clone()
    v4 = v1.unfold(0, 1, 2)
    v5 = v4[:1:2, 3:5:3, :]
    t1.unsqueeze_(0)
    return (v2, v5, t1, v2, lst, dct)

INPUTS = [('t0', (4, 6)), ('t1', (2, 3, 4))]
ALIASED = None
DTYPE = 'i64'
DYNAMIC = None
DEVICE = 'cuda'
BACKEND = 'inductor'
SEED = 71001416
CALLS = 2

def diff(a, b, path, out, depth=0):
    if depth > 6:
        return
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            out.append('%s: type %s vs %s' % (path, type(a).__name__, type(b).__name__))
            return
        if tuple(a.shape) != tuple(b.shape):
            out.append('%s: shape %s vs %s' % (path, tuple(a.shape), tuple(b.shape)))
            return
        if a.dtype != b.dtype:
            out.append('%s: dtype %s vs %s' % (path, a.dtype, b.dtype))
            return
        ac, bc = a.detach().cpu(), b.detach().cpu()
        if not torch.equal(ac, bc):
            fa, fb = ac.flatten(), bc.flatten()
            ne = fa != fb
            bad = ne.nonzero().flatten()[:5].tolist()
            out.append('%s: %d/%d elems differ at %s: eager %s vs compiled %s' % (
                path, int(ne.sum()), fa.numel(), bad, [fa[i].item() for i in bad], [fb[i].item() for i in bad]))
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append('%s: len %d vs %d' % (path, len(a), len(b)))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, '%s[%d]' % (path, i), out, depth + 1)
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            out.append('%s: keys %s vs %s' % (path, sorted(map(str, a)), sorted(map(str, b))))
            return
        for kk in a:
            diff(a[kk], b[kk], '%s[%r]' % (path, kk), out, depth + 1)
        return
    try:
        eq = bool(a == b)
    except Exception:
        eq = False
    if not eq:
        out.append('%s: %r vs %r' % (path, a, b))

def flatten_tensors(obj, acc, depth=0):
    if depth > 6:
        return
    if isinstance(obj, torch.Tensor):
        acc.append(obj)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            flatten_tensors(x, acc, depth + 1)
    elif isinstance(obj, dict):
        for x in obj.values():
            flatten_tensors(x, acc, depth + 1)

def probe(worlds):
    log = []
    for w in worlds:
        acc = []
        flatten_tensors(w['out'], acc)
        for i, t in enumerate(acc):
            if t.dtype not in (torch.int64, torch.float64, torch.int32):
                log.append('skip')
                continue
            try:
                t.add_(1000 * (i + 1))
                log.append('ok')
            except Exception as e:
                log.append('err:%s' % type(e).__name__)
    return log


def make_args(seed):
    gen = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, shape in INPUTS:
        t = torch.randint(-8, 9, tuple(shape), generator=gen, dtype=torch.int64)
        if DTYPE == 'f64':
            t = t.to(torch.float64)
        tensors[name] = t.to(DEVICE)
    args = [tensors[n] for n, _ in INPUTS]
    if ALIASED:
        args.append(eval(ALIASED[1], {'torch': torch, 't0': tensors['t0']}))
    args += [(seed % 5) - 2, [], {}]
    return args


def run(compiled):
    f = fn
    if compiled:
        torch._dynamo.reset()
        f = torch.compile(fn, backend=BACKEND, dynamic=DYNAMIC)
    worlds = []
    for c in range(CALLS):
        args = make_args(SEED * 7919 + c)
        worlds.append({'out': f(*args), 'args': args})
    return worlds


def compare(E, C, prefix=''):
    diffs = []
    for i, (e, c) in enumerate(zip(E, C)):
        diff(e['out'], c['out'], '%scall%d.out' % (prefix, i), diffs)
        for j, (x, y) in enumerate(zip(e['args'], c['args'])):
            diff(x, y, '%scall%d.arg%d' % (prefix, i, j), diffs)
    return diffs


if __name__ == '__main__':
    print('torch', torch.__version__)
    E = run(False)
    C = run(True)
    d1 = compare(E, C)
    print('primary diffs:', len(d1))
    for d in d1:
        print('  ', d)
    pe, pc = probe(E), probe(C)
    d2 = compare(E, C, 'probe.')
    print('probe log eager   :', pe)
    print('probe log compiled:', pc)
    print('alias-probe diffs:', len(d2))
    for d in d2:
        print('  ', d)
