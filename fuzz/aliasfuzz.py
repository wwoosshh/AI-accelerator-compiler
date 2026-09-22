# -*- coding: utf-8 -*-
"""
aliasfuzz.py : torch.compile 차등 테스터 (alias / view / in-place / Python side-effect 특화)

사용 예
  python aliasfuzz.py --backend aot_eager --device cpu  --minutes 15 --out results/aot_eager_cpu
  python aliasfuzz.py --backend inductor  --device cuda --minutes 25 --out results/inductor_cuda

오라클 (모두 정수값 데이터만 사용 -> 허용오차 0 으로 비교)
  1) 반환값 비교
  2) 함수가 in-place 로 바꾼 입력 텐서의 최종 상태 비교 (입력 변이 write-back)
  3) 인자로 넘긴 list / dict 에 대한 Python 부작용 비교 (Dynamo side-effect replay)
  4) 별칭 프로브: 반환 텐서를 in-place 로 건드린 뒤 다른 반환값/입력으로 전파되는 양상이 eager 와 같은지
  5) 같은 컴파일 결과를 데이터와 int 인자를 바꿔 2회 호출 (가드/캐시 버그)
"""
import argparse
import inspect
import json
import os
import random
import sys
import time
import traceback

import torch

BASE_SHAPES = [(4, 6), (6, 4), (2, 3, 4), (24,), (3, 8), (2, 12), (4, 3, 2), (2, 2, 6)]


def prod(t):
    p = 1
    for x in t:
        p *= x
    return p


def shape_txt(shape):
    if len(shape) == 1:
        return '(%d,)' % shape[0]
    return '(%s)' % ', '.join(str(s) for s in shape)


class Var:
    def __init__(self, name, shape, dtype, writable=True, contiguous=True, is_input=False, root=None):
        self.name = name
        self.shape = tuple(shape)
        self.dtype = dtype            # 'i64' | 'f64' | 'i32'
        self.writable = writable      # False: expand / overlapping unfold (쓰면 eager 오류)
        self.contiguous = contiguous  # 보수적 추정
        self.is_input = is_input
        self.root = root if root is not None else name  # 저장소 뿌리(추정)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def numel(self):
        return prod(self.shape)


class Stmt:
    def __init__(self, code, kind):
        self.code = code
        self.kind = kind


class Program:
    def __init__(self, inputs, stmts, returns, dtype, dynamic, aliased_input):
        self.inputs = inputs                # [(name, shape)]
        self.stmts = stmts                  # [Stmt]
        self.returns = returns              # [expr]
        self.dtype = dtype                  # 'i64' | 'f64'
        self.dynamic = dynamic              # None | True
        self.aliased_input = aliased_input  # None | (name, expr_of_t0)

    def clone(self, stmts=None, returns=None, aliased_input='keep', dynamic='keep'):
        return Program(self.inputs,
                       list(self.stmts if stmts is None else stmts),
                       list(self.returns if returns is None else returns),
                       self.dtype,
                       self.dynamic if dynamic == 'keep' else dynamic,
                       self.aliased_input if aliased_input == 'keep' else aliased_input)

    def arg_names(self):
        names = [n for n, _ in self.inputs]
        if self.aliased_input:
            names.append(self.aliased_input[0])
        return names + ['k', 'lst', 'dct']

    def source(self):
        lines = ['def fn(%s):' % ', '.join(self.arg_names())]
        for s in self.stmts:
            lines.append('    ' + s.code)
        rets = ', '.join(self.returns)
        if len(self.returns) == 1:
            rets += ','
        lines.append('    return (%s)' % rets)
        return '\n'.join(lines) + '\n'

    def kinds(self):
        return sorted(set(s.kind for s in self.stmts))


# ----------------------------------------------------------------------------
# 프로그램 생성기
# ----------------------------------------------------------------------------
class Gen:
    def __init__(self, rng, dtype):
        self.rng = rng
        self.dtype = dtype
        self.vars = []
        self.stmts = []
        self.counter = 0
        self.inputs = []
        self.aliased_input = None

    # ---- helpers ----
    def fresh(self, prefix='v'):
        self.counter += 1
        return '%s%d' % (prefix, self.counter)

    def emit(self, code, kind):
        self.stmts.append(Stmt(code, kind))

    def add(self, v):
        self.vars.append(v)
        return v

    def pick(self, pred=None, recent=0.5):
        c = [v for v in self.vars if pred is None or pred(v)]
        if not c:
            return None
        if len(c) > 4 and self.rng.random() < recent:
            c = c[-4:]
        return self.rng.choice(c)

    def identity_twins(self, v):
        """v 와 같은 텐서 객체를 가리키는 다른 변수. 별칭 입력이 `ta = t0` 형태(뷰가 아니라 같은 객체)이면
        transpose_ 같은 in-place 메타데이터 연산이 두 이름의 형상을 함께 바꾸므로 생성기도 같이 갱신해야 한다
        (E2 1차 실행에서 이를 놓쳐 범위 밖 인덱스가 만들어졌고, CUDA 에서는 device-side assert 로 컨텍스트가 오염됐다)."""
        if self.aliased_input and self.aliased_input[1] == 't0' and v.name in ('t0', 'ta'):
            return [w for w in self.vars if w.name in ('t0', 'ta') and w is not v]
        return []

    def other(self, v, diff_root=False):
        """같은 shape/dtype 의 다른 변수. diff_root=True 면 저장소 뿌리가 다른 것만 (in-place 피연산자용:
        대상과 부분적으로 겹치는 메모리를 읽으면서 쓰는 것은 PyTorch 계약상 정의되지 않은 동작이라 오탐이 됨)."""
        c = [w for w in self.vars if w.shape == v.shape and w.dtype == v.dtype and w.name != v.name]
        if diff_root:
            c = [w for w in c if w.root != v.root]
        elif self.rng.random() < 0.7:
            c2 = [w for w in c if w.root != v.root]
            if c2:
                c = c2
        return self.rng.choice(c) if c else None

    def scalar(self):
        if self.rng.random() < 0.2:
            return 'k'
        return str(self.rng.randint(-5, 5))

    def rand_slice(self, n):
        """(text, start, stop, step, length)"""
        if n == 1:
            return ':', 0, 1, 1, 1
        step = self.rng.choice([1, 1, 1, 2, 2, 3])
        start = self.rng.randint(0, n - 1)
        stop = self.rng.randint(start + 1, n)
        length = (stop - start + step - 1) // step
        a = '' if (start == 0 and self.rng.random() < 0.5) else str(start)
        b = '' if (stop == n and self.rng.random() < 0.5) else str(stop)
        if b and stop < n and self.rng.random() < 0.2:
            b = str(stop - n)
        txt = '%s:%s' % (a, b)
        if step != 1:
            txt += ':%d' % step
        return txt, start, stop, step, length

    # ---- view creators ----
    def g_view_slice(self):
        a = self.pick(lambda v: v.ndim >= 1)
        if a is None:
            return False
        parts, shape, contig = [], [], a.contiguous
        for i, n in enumerate(a.shape):
            r = self.rng.random()
            remaining = a.ndim - i - 1
            if r < 0.35:
                parts.append(':')
                shape.append(n)
            elif r < 0.5 and (len(shape) + remaining) >= 1:
                parts.append(str(self.rng.randint(-n, n - 1)))
                contig = False
            else:
                txt, _, _, _, ln = self.rand_slice(n)
                parts.append(txt)
                shape.append(ln)
                if txt != ':':
                    contig = False
        if not shape:
            return False
        name = self.fresh()
        self.emit('%s = %s[%s]' % (name, a.name, ', '.join(parts)), 'view_slice')
        self.add(Var(name, shape, a.dtype, a.writable, contig, root=a.root))
        return True

    def g_view_transpose(self):
        a = self.pick(lambda v: v.ndim >= 2)
        if a is None:
            return False
        name = self.fresh()
        r = self.rng.random()
        if a.ndim == 2 and r < 0.35:
            code = '%s = %s.t()' % (name, a.name)
            shape = (a.shape[1], a.shape[0])
        elif r < 0.7:
            d0, d1 = self.rng.sample(range(a.ndim), 2)
            code = '%s = %s.transpose(%d, %d)' % (name, a.name, d0, d1)
            shape = list(a.shape)
            shape[d0], shape[d1] = shape[d1], shape[d0]
        else:
            perm = list(range(a.ndim))
            self.rng.shuffle(perm)
            code = '%s = %s.permute(%s)' % (name, a.name, ', '.join(map(str, perm)))
            shape = [a.shape[p] for p in perm]
        self.emit(code, 'view_transpose')
        self.add(Var(name, shape, a.dtype, a.writable, False, root=a.root))
        return True

    def g_view_reshape(self):
        a = self.pick(lambda v: v.numel >= 2)
        if a is None:
            return False
        name = self.fresh()
        cands = [s for s in BASE_SHAPES if prod(s) == a.numel and s != a.shape]
        if cands and self.rng.random() < 0.7:
            shape = self.rng.choice(cands)
        else:
            shape = (a.numel,) if a.shape != (a.numel,) else (1, a.numel)
        r = self.rng.random()
        if a.contiguous and r < 0.5:
            code = '%s = %s.view(%s)' % (name, a.name, shape_txt(shape))
        elif r < 0.65:
            code = '%s = %s.flatten()' % (name, a.name)
            shape = (a.numel,)
        else:
            code = '%s = %s.reshape(%s)' % (name, a.name, shape_txt(shape))
        self.emit(code, 'view_reshape')
        self.add(Var(name, shape, a.dtype, a.writable, a.contiguous, root=a.root))
        return True

    def g_view_squeeze(self):
        a = self.pick()
        if a is None:
            return False
        name = self.fresh()
        ones = [i for i, s in enumerate(a.shape) if s == 1]
        if ones and self.rng.random() < 0.5:
            d = self.rng.choice(ones)
            code = '%s = %s.squeeze(%d)' % (name, a.name, d)
            shape = a.shape[:d] + a.shape[d + 1:]
        else:
            d = self.rng.randint(0, a.ndim)
            code = '%s = %s.unsqueeze(%d)' % (name, a.name, d)
            shape = a.shape[:d] + (1,) + a.shape[d:]
        self.emit(code, 'view_squeeze')
        self.add(Var(name, shape, a.dtype, a.writable, a.contiguous, root=a.root))
        return True

    def g_view_select(self):
        a = self.pick(lambda v: v.ndim >= 2)
        if a is None:
            return False
        d = self.rng.randrange(a.ndim)
        i = self.rng.randint(0, a.shape[d] - 1)
        name = self.fresh()
        if self.rng.random() < 0.5:
            code = '%s = %s.select(%d, %d)' % (name, a.name, d, i)
            shape = a.shape[:d] + a.shape[d + 1:]
        else:
            ln = self.rng.randint(1, a.shape[d] - i)
            code = '%s = %s.narrow(%d, %d, %d)' % (name, a.name, d, i, ln)
            shape = list(a.shape)
            shape[d] = ln
        self.emit(code, 'view_select')
        self.add(Var(name, shape, a.dtype, a.writable, False, root=a.root))
        return True

    def g_view_diag_unfold_expand(self):
        a = self.pick(lambda v: v.ndim >= 1)
        if a is None:
            return False
        name = self.fresh()
        r = self.rng.random()
        if a.ndim == 2 and r < 0.35:
            code = '%s = %s.diagonal()' % (name, a.name)
            self.emit(code, 'view_diagonal')
            self.add(Var(name, (min(a.shape),), a.dtype, a.writable, False, root=a.root))
            return True
        if r < 0.7:
            d = self.rng.randrange(a.ndim)
            n = a.shape[d]
            if n < 2:
                return False
            size = self.rng.randint(1, n)
            step = self.rng.randint(1, n)
            cnt = (n - size) // step + 1
            shape = list(a.shape)
            shape[d] = cnt
            shape.append(size)
            code = '%s = %s.unfold(%d, %d, %d)' % (name, a.name, d, size, step)
            self.emit(code, 'view_unfold')
            self.add(Var(name, shape, a.dtype, a.writable and step >= size, False, root=a.root))
            return True
        m = self.rng.randint(2, 3)
        code = '%s = %s.unsqueeze(0).expand(%d, %s)' % (name, a.name, m, ', '.join(map(str, a.shape)))
        self.emit(code, 'view_expand')
        self.add(Var(name, (m,) + a.shape, a.dtype, False, False, root=a.root))
        return True

    def g_view_dtype(self):
        a = self.pick(lambda v: v.contiguous and v.ndim >= 1 and v.dtype in ('i64', 'i32'))
        if a is None:
            return False
        name = self.fresh()
        if a.dtype == 'i64':
            code = '%s = %s.view(torch.int32)' % (name, a.name)
            shape = a.shape[:-1] + (a.shape[-1] * 2,)
            dt = 'i32'
        else:
            if a.shape[-1] % 2:
                return False
            code = '%s = %s.view(torch.int64)' % (name, a.name)
            shape = a.shape[:-1] + (a.shape[-1] // 2,)
            dt = 'i64'
        self.emit(code, 'view_dtype')
        self.add(Var(name, shape, dt, a.writable, True, root=a.root))
        return True

    def g_noop_alias(self):
        a = self.pick()
        if a is None:
            return False
        name = self.fresh()
        # '%s + 0' / '%s * 1' 은 알려진 이슈 #197893 (입력 자체를 반환) 재발견만 반복하므로 제외
        forms = ['%s.contiguous()', '%s.detach()', '%s.view_as(%s)', '%s[...]',
                 '%s[:]' if a.ndim else '%s[...]', 'torch.ops.aten.alias(%s)', '%s.to(%s.dtype)',
                 '%s.reshape(%s.shape)', '%s.clone()']
        f = self.rng.choice(forms)
        expr = f.replace('%s', a.name)
        is_alias = f != '%s.clone()'
        self.emit('%s = %s' % (name, expr), 'noop_alias' if is_alias else 'noop_arith')
        if is_alias:
            self.add(Var(name, a.shape, a.dtype, a.writable, a.contiguous, root=a.root))
        else:
            self.add(Var(name, a.shape, a.dtype, True, True, root=name))
        return True

    # ---- functional ----
    def g_functional(self):
        a = self.pick(lambda v: v.dtype in ('i64', 'f64', 'i32'))
        if a is None:
            return False
        name = self.fresh()
        r = self.rng.random()
        shape = a.shape
        b = self.other(a)
        if r < 0.25:
            op = self.rng.choice(['+', '-', '+', '*'])
            if b is not None and self.rng.random() < 0.7:
                rhs = b.name
                if op == '*' and self.rng.random() < 0.7:
                    op = '+'
            else:
                # 'x + 0' / 'x * 1' 은 알려진 이슈 #197893 재발견이라 상수에서 제외
                rhs = str(self.rng.choice([-3, -2, -1, 1, 2, 3])) if op != '*' else str(self.rng.choice([-1, 0, 2]))
            code = '%s = %s %s %s' % (name, a.name, op, rhs)
        elif r < 0.35 and b is not None:
            code = '%s = torch.where(%s > %d, %s, %s)' % (name, a.name, self.rng.randint(-2, 2), a.name, b.name)
        elif r < 0.45 and b is not None:
            code = '%s = torch.%s(%s, %s)' % (name, self.rng.choice(['maximum', 'minimum']), a.name, b.name)
        elif r < 0.55:
            code = '%s = %s.%s()' % (name, a.name, self.rng.choice(['neg', 'abs']))
        elif r < 0.62:
            code = '%s = %s.clamp(%d, %d)' % (name, a.name, self.rng.randint(-4, 0), self.rng.randint(1, 4))
        elif r < 0.72 and a.ndim >= 1:
            d = self.rng.randrange(a.ndim)
            if self.rng.random() < 0.5:
                code = '%s = torch.roll(%s, %d, %d)' % (name, a.name, self.rng.randint(1, 3), d)
            else:
                code = '%s = %s.flip(%d)' % (name, a.name, d)
        elif r < 0.8 and a.ndim >= 1 and a.dtype != 'i32':
            d = self.rng.randrange(a.ndim)
            code = '%s = %s.cumsum(%d)' % (name, a.name, d)
        elif r < 0.9 and a.ndim >= 1 and a.dtype != 'i32':
            d = self.rng.randrange(a.ndim)
            keep = self.rng.random() < 0.5
            code = '%s = %s.sum(%d, keepdim=%s)' % (name, a.name, d, keep)
            shape = a.shape[:d] + ((1,) if keep else ()) + a.shape[d + 1:]
        elif b is not None and a.ndim >= 1:
            d = self.rng.randrange(a.ndim)
            code = '%s = torch.cat([%s, %s], %d)' % (name, a.name, b.name, d)
            shape = list(a.shape)
            shape[d] *= 2
        else:
            code = '%s = %s * 2' % (name, a.name)
        self.emit(code, 'functional')
        self.add(Var(name, shape, a.dtype, True, True, root=name))
        return True

    # ---- in-place ----
    def g_inplace(self):
        t = self.pick(lambda v: v.writable and v.dtype in ('i64', 'f64', 'i32'), recent=0.3)
        if t is None:
            return False
        b = self.other(t, diff_root=True)
        r = self.rng.random()
        if r < 0.3:
            op = self.rng.choice(['add_', 'sub_', 'add_', 'mul_'])
            if b is not None and self.rng.random() < 0.6:
                arg = b.name
                if op == 'mul_' and self.rng.random() < 0.7:
                    op = 'add_'
            elif b is not None and t.ndim >= 1 and self.rng.random() < 0.5:
                d = self.rng.randrange(t.ndim)
                arg = '%s.narrow(%d, 0, 1)' % (b.name, d)
            else:
                arg = self.scalar() if op != 'mul_' else str(self.rng.choice([-1, 0, 1, 2]))
            code = '%s.%s(%s)' % (t.name, op, arg)
        elif r < 0.45:
            if b is None or self.rng.random() < 0.3:
                code = '%s.fill_(%s)' % (t.name, self.scalar())
            else:
                code = '%s.copy_(%s)' % (t.name, b.name)
        elif r < 0.55:
            code = '%s.%s()' % (t.name, self.rng.choice(['zero_', 'neg_', 'abs_']))
        elif r < 0.65:
            code = '%s.clamp_(%d, %d)' % (t.name, self.rng.randint(-4, 0), self.rng.randint(1, 4))
        elif r < 0.8:
            m = b if (b is not None and self.rng.random() < 0.7) else t
            code = '%s.masked_fill_(%s > %d, %s)' % (t.name, m.name, self.rng.randint(-2, 2), self.scalar())
        elif r < 0.9 and t.ndim >= 1:
            code = '%s.add_(%s.flip(%d))' % (t.name, t.name, self.rng.randrange(t.ndim))
        else:
            code = '%s.add_(%s)' % (t.name, t.name)
        self.emit(code, 'inplace')
        return True

    def g_setitem(self):
        t = self.pick(lambda v: v.writable and v.dtype in ('i64', 'f64', 'i32') and v.ndim >= 1, recent=0.3)
        if t is None:
            return False
        b = self.other(t, diff_root=True)
        r = self.rng.random()
        if r < 0.4:
            parts = []
            for n in t.shape:
                parts.append(':' if self.rng.random() < 0.4 else self.rand_slice(n)[0])
            sl = ', '.join(parts)
            rr = self.rng.random()
            if b is not None and rr < 0.4:
                code = '%s[%s] = %s[%s]' % (t.name, sl, b.name, sl)
            elif rr < 0.55 and t.shape[0] >= 2:
                # 겹치는 자기 대입: 소스를 clone 해야 정의된 동작 (이슈 #197829 패턴)
                rest = ''.join([', :'] * (t.ndim - 1))
                if self.rng.random() < 0.5:
                    code = '%s[1:%s] = %s[:-1%s].clone()' % (t.name, rest, t.name, rest)
                else:
                    code = '%s[:-1%s] = %s[1:%s].clone()' % (t.name, rest, t.name, rest)
            elif rr < 0.75:
                code = '%s[%s] += %s' % (t.name, sl, self.scalar())
            else:
                code = '%s[%s] = %s' % (t.name, sl, self.scalar())
        elif r < 0.6:
            n = t.shape[0]
            kk = self.rng.randint(1, min(3, n))
            vals = self.rng.sample(range(n), kk)
            idx = self.fresh('idx')
            self.emit('%s = torch.tensor(%s, device=%s.device)' % (idx, vals, t.name), 'aux')
            if self.rng.random() < 0.5:
                code = '%s[%s] = %s' % (t.name, idx, self.scalar())
            else:
                code = '%s[%s] += %s' % (t.name, idx, self.scalar())
        elif r < 0.8:
            m = b if (b is not None and self.rng.random() < 0.7) else t
            code = '%s[%s > %d] = %s' % (t.name, m.name, self.rng.randint(-2, 2), self.scalar())
        else:
            i = self.rng.randint(-t.shape[0], t.shape[0] - 1)
            if b is not None and self.rng.random() < 0.5:
                j = self.rng.randint(0, t.shape[0] - 1)
                code = '%s[%d] = %s[%d]' % (t.name, i, b.name, j)
            else:
                code = '%s[%d] = %s' % (t.name, i, self.scalar())
        self.emit(code, 'setitem')
        return True

    def g_index_inplace(self):
        t = self.pick(lambda v: v.writable and v.dtype in ('i64', 'f64') and v.ndim >= 1, recent=0.3)
        if t is None:
            return False
        d = self.rng.randrange(t.ndim)
        n = t.shape[d]
        kk = self.rng.randint(1, min(3, n))
        vals = self.rng.sample(range(n), kk)
        idx = self.fresh('idx')
        self.emit('%s = torch.tensor(%s, device=%s.device)' % (idx, vals, t.name), 'aux')
        b = self.other(t, diff_root=True)
        r = self.rng.random()
        if b is not None and r < 0.35:
            code = '%s.index_add_(%d, %s, %s.narrow(%d, 0, %d))' % (t.name, d, idx, b.name, d, kk)
        elif b is not None and r < 0.6:
            code = '%s.index_copy_(%d, %s, %s.narrow(%d, 0, %d))' % (t.name, d, idx, b.name, d, kk)
        elif b is not None and r < 0.8:
            code = '%s.scatter_add_(%d, (%s.abs() %% %d).long(), %s)' % (t.name, d, b.name, n, b.name)
        else:
            code = '%s.index_fill_(%d, %s, %s)' % (t.name, d, idx, self.scalar())
        self.emit(code, 'index_inplace')
        return True

    def g_inplace_meta(self):
        t = self.pick(lambda v: v.ndim >= 1 and v.dtype in ('i64', 'f64'))
        if t is None:
            return False
        r = self.rng.random()
        if t.ndim >= 2 and r < 0.4:
            d0, d1 = self.rng.sample(range(t.ndim), 2)
            code = '%s.transpose_(%d, %d)' % (t.name, d0, d1)
            s = list(t.shape)
            s[d0], s[d1] = s[d1], s[d0]
            t.shape = tuple(s)
            t.contiguous = False
        elif t.ndim == 2 and r < 0.55:
            code = '%s.t_()' % t.name
            t.shape = (t.shape[1], t.shape[0])
            t.contiguous = False
        elif r < 0.8:
            d = self.rng.randint(0, t.ndim)
            code = '%s.unsqueeze_(%d)' % (t.name, d)
            t.shape = t.shape[:d] + (1,) + t.shape[d:]
        else:
            ones = [i for i, s in enumerate(t.shape) if s == 1]
            if not ones:
                return False
            d = self.rng.choice(ones)
            code = '%s.squeeze_(%d)' % (t.name, d)
            t.shape = t.shape[:d] + t.shape[d + 1:]
        for tw in self.identity_twins(t):
            tw.shape = t.shape
            tw.contiguous = t.contiguous
        self.emit(code, 'inplace_meta')
        return True

    def g_scatter_fn(self):
        a = self.pick(lambda v: v.dtype in ('i64', 'f64') and v.ndim >= 1)
        if a is None:
            return False
        b = self.other(a)
        if b is None:
            return False
        d = self.rng.randrange(a.ndim)
        n = a.shape[d]
        name = self.fresh()
        r = self.rng.random()
        if r < 0.45:
            _, start, stop, step, _ = self.rand_slice(n)
            parts = [':'] * a.ndim
            parts[d] = '%d:%d:%d' % (start, stop, step)
            code = '%s = torch.slice_scatter(%s, %s[%s], dim=%d, start=%d, end=%d, step=%d)' % (
                name, a.name, b.name, ', '.join(parts), d, start, stop, step)
        elif r < 0.8 or a.ndim != 2:
            i = self.rng.randint(0, n - 1)
            code = '%s = torch.select_scatter(%s, %s.select(%d, %d), %d, %d)' % (name, a.name, b.name, d, i, d, i)
        else:
            code = '%s = torch.diagonal_scatter(%s, %s.diagonal())' % (name, a.name, b.name)
        self.emit(code, 'scatter_fn')
        self.add(Var(name, a.shape, a.dtype, True, True, root=name))
        return True

    def g_side_effect(self):
        v = self.pick()
        if v is None:
            return False
        r = self.rng.random()
        if r < 0.35:
            code = 'lst.append(%s)' % v.name
        elif r < 0.6:
            code = "dct['%s'] = %s" % (self.fresh('key'), v.name)
        elif r < 0.75:
            code = "dct['n'] = dct.get('n', 0) + 1"
        elif r < 0.9 and v.ndim >= 1:
            code = 'lst.append(%s.shape[%d] + k)' % (v.name, self.rng.randrange(v.ndim))
        elif r < 0.95:
            code = 'lst.append(int(%s.sum()))' % v.name
        else:
            code = 'lst.append(%s.numel())' % v.name
        self.emit(code, 'side_effect')
        return True


def generate(rng, dtype_mode, n_stmts):
    dtype = dtype_mode if dtype_mode in ('i64', 'f64') else rng.choice(['i64', 'i64', 'f64'])
    g = Gen(rng, dtype)
    n_in = rng.randint(2, 3)
    for i in range(n_in):
        shape = rng.choice(BASE_SHAPES)
        g.inputs.append(('t%d' % i, shape))
        g.add(Var('t%d' % i, shape, dtype, True, True, True, root='t%d' % i))
    if rng.random() < 0.3:
        s0 = tuple(g.inputs[0][1])
        forms = [('t0.view(-1)', (prod(s0),), True), ('t0[1:]', (s0[0] - 1,) + s0[1:], False), ('t0', s0, True)]
        if len(s0) >= 2:
            forms.append(('t0.transpose(0, 1)', (s0[1], s0[0]) + s0[2:], False))
        expr, shape, contig = rng.choice(forms)
        g.aliased_input = ('ta', expr)
        g.add(Var('ta', shape, dtype, True, contig, True, root='t0'))
    gens = [(g.g_view_slice, 3), (g.g_view_transpose, 2), (g.g_view_reshape, 2), (g.g_view_squeeze, 1),
            (g.g_view_select, 1), (g.g_view_diag_unfold_expand, 1), (g.g_view_dtype, 1), (g.g_noop_alias, 2),
            (g.g_functional, 3), (g.g_inplace, 4), (g.g_setitem, 3), (g.g_index_inplace, 2),
            (g.g_inplace_meta, 1), (g.g_scatter_fn, 2), (g.g_side_effect, 1)]
    funcs = [f for f, _ in gens]
    weights = [w for _, w in gens]
    tries = 0
    while len(g.stmts) < n_stmts and tries < n_stmts * 10:
        tries += 1
        rng.choices(funcs, weights)[0]()
    rets = [rng.choice(g.vars).name for _ in range(rng.randint(2, 4))]
    if rng.random() < 0.6:
        rets.append(rng.choice([n for n, _ in g.inputs]))
    if rng.random() < 0.3:
        rets.append(rets[0])
    rets += ['lst', 'dct']
    dynamic = True if rng.random() < 0.35 else None
    return Program(g.inputs, g.stmts, rets, dtype, dynamic, g.aliased_input)


# ----------------------------------------------------------------------------
# 실행 / 비교
# ----------------------------------------------------------------------------
def make_args(prog, device, seed):
    gen = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, shape in prog.inputs:
        t = torch.randint(-8, 9, tuple(shape), generator=gen, dtype=torch.int64)
        if prog.dtype == 'f64':
            t = t.to(torch.float64)
        tensors[name] = t.to(device)
    args = [tensors[n] for n, _ in prog.inputs]
    if prog.aliased_input:
        args.append(eval(prog.aliased_input[1], {'torch': torch, 't0': tensors['t0']}))
    args += [(seed % 5) - 2, [], {}]
    return args


def build_fn(prog):
    ns = {'torch': torch}
    exec(compile(prog.source(), '<aliasfuzz>', 'exec'), ns)
    return ns['fn']


def run_world(prog, device, seed, backend, calls):
    fn = build_fn(prog)
    if backend is not None:
        torch._dynamo.reset()
        fn = torch.compile(fn, backend=backend, dynamic=prog.dynamic)
    worlds = []
    for c in range(calls):
        args = make_args(prog, device, seed * 7919 + c)
        out = fn(*args)
        worlds.append({'out': out, 'args': args})
    return worlds


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


def compare_worlds(E, C, prefix=''):
    diffs = []
    for i, (e, c) in enumerate(zip(E, C)):
        diff(e['out'], c['out'], '%scall%d.out' % (prefix, i), diffs)
        for j, (x, y) in enumerate(zip(e['args'], c['args'])):
            diff(x, y, '%scall%d.arg%d' % (prefix, i, j), diffs)
    return diffs


def evaluate(prog, device, backend, seed, calls):
    """status: invalid | compile_error | value_diff | alias_diff | pass"""
    try:
        E = run_world(prog, device, seed, None, calls)
    except Exception as e:
        return 'invalid', '%s: %s' % (type(e).__name__, str(e)[:200])
    try:
        C = run_world(prog, device, seed, backend, calls)
    except Exception as e:
        return 'compile_error', '%s: %s\n%s' % (type(e).__name__, str(e)[:600], traceback.format_exc()[-2000:])
    diffs = compare_worlds(E, C)
    if diffs:
        return 'value_diff', diffs
    pe, pc = probe(E), probe(C)
    diffs = compare_worlds(E, C, 'probe.')
    if pe != pc:
        diffs.insert(0, 'probe log differs: eager %s vs compiled %s' % (pe, pc))
    if diffs:
        return 'alias_diff', diffs
    return 'pass', None


def cuda_healthy():
    """eager 쪽 device-side assert(예: 범위 밖 인덱스) 뒤에는 CUDA 컨텍스트가 회복 불가능하게 오염되어
    이후 모든 프로그램이 'invalid' 로 집계된다(E2 1차 실행: 1228번째 이후 333만 회). 그 상태를 감지한다."""
    try:
        torch.cuda.synchronize()
        (torch.ones(2, device='cuda') + 1).sum().item()
        return True
    except Exception:
        return False


def minimize(prog, device, backend, seed, calls, budget):
    cur = prog
    used = 0

    def still_bad(p):
        st, _ = evaluate(p, device, backend, seed, calls)
        return st in ('value_diff', 'alias_diff')

    improved = True
    while improved and used < budget:
        improved = False
        for i in range(len(cur.stmts) - 1, -1, -1):
            if used >= budget:
                break
            trial = cur.clone(stmts=cur.stmts[:i] + cur.stmts[i + 1:])
            used += 1
            if still_bad(trial):
                cur = trial
                improved = True
        for i in range(len(cur.returns) - 3, -1, -1):
            if used >= budget or len(cur.returns) <= 3:
                break
            trial = cur.clone(returns=cur.returns[:i] + cur.returns[i + 1:])
            used += 1
            if still_bad(trial):
                cur = trial
                improved = True
        if cur.aliased_input and used < budget:
            trial = cur.clone(aliased_input=None)
            used += 1
            if still_bad(trial):
                cur = trial
                improved = True
        if cur.dynamic and used < budget:
            trial = cur.clone(dynamic=None)
            used += 1
            if still_bad(trial):
                cur = trial
                improved = True
    return cur, used


REPRO_MAIN = r'''
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
'''


def program_from_repro(path):
    """저장된 repro .py 에서 Program 과 상수(BACKEND/DEVICE/SEED/CALLS)를 복원 (생성기가 바뀌어도 최소화 가능)."""
    lines = open(path, encoding='utf-8').read().splitlines()
    i0 = next(i for i, l in enumerate(lines) if l.startswith('def fn('))
    i1 = next(i for i in range(i0, len(lines)) if lines[i].startswith('    return ('))
    stmts = [Stmt(l[4:], 'x') for l in lines[i0 + 1:i1] if l.strip()]
    ret = lines[i1].strip()[len('return ('):-1]
    returns = [r.strip() for r in ret.split(',') if r.strip()]
    consts = {}
    for l in lines:
        for key in ('INPUTS', 'ALIASED', 'DTYPE', 'DYNAMIC', 'DEVICE', 'BACKEND', 'SEED', 'CALLS'):
            if l.startswith(key + ' = '):
                consts[key] = eval(l[len(key) + 3:])
    aliased = tuple(consts['ALIASED']) if consts.get('ALIASED') else None
    prog = Program([(n, tuple(s)) for n, s in consts['INPUTS']], stmts, returns, consts['DTYPE'],
                   consts['DYNAMIC'], aliased)
    return prog, consts


def repro_source(prog, device, backend, seed, calls, kind):
    parts = ['# -*- coding: utf-8 -*-',
             '# aliasfuzz repro | status=%s backend=%s device=%s seed=%d' % (kind, backend, device, seed),
             'import torch', '', prog.source(),
             'INPUTS = %r' % (prog.inputs,), 'ALIASED = %r' % (prog.aliased_input,), 'DTYPE = %r' % prog.dtype,
             'DYNAMIC = %r' % prog.dynamic, 'DEVICE = %r' % device, 'BACKEND = %r' % backend,
             'SEED = %d' % seed, 'CALLS = %d' % calls, '',
             inspect.getsource(diff), inspect.getsource(flatten_tensors), inspect.getsource(probe), REPRO_MAIN]
    return '\n'.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', default='aot_eager')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--minutes', type=float, default=10)
    ap.add_argument('--out', default='results/run')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--dtype', default='mix', choices=['i64', 'f64', 'mix'])
    ap.add_argument('--min-stmts', type=int, default=3)
    ap.add_argument('--max-stmts', type=int, default=10)
    ap.add_argument('--calls', type=int, default=2)
    ap.add_argument('--minimize', type=int, default=6)
    ap.add_argument('--min-budget', type=int, default=40)
    ap.add_argument('--max-cases', type=int, default=60)
    ap.add_argument('--minimize-file', action='append', default=[],
                    help='저장된 case_XXX.py 를 읽어 최소화만 수행 (backend/device/seed 는 파일의 값 사용)')
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    if args.minimize_file:
        torch._dynamo.config.cache_size_limit = 64
        for path in args.minimize_file:
            prog, c = program_from_repro(path)
            backend, device, seed, calls = c['BACKEND'], c['DEVICE'], c['SEED'], c['CALLS']
            st0, info0 = evaluate(prog, device, backend, seed, calls)
            print('== %s | torch %s | %s/%s | initial status: %s (%d stmts)' % (
                path, torch.__version__, backend, device, st0, len(prog.stmts)), flush=True)
            if st0 not in ('value_diff', 'alias_diff'):
                print('   not divergent any more, skipping. info:', str(info0)[:300])
                continue
            mprog, used = minimize(prog, device, backend, seed, calls, args.min_budget)
            st, info = evaluate(mprog, device, backend, seed, calls)
            outp = path[:-3] + '.min.py'
            with open(outp, 'w', encoding='utf-8') as f:
                f.write(repro_source(mprog, device, backend, seed, calls, st))
            print('   -> %d stmts, status=%s, checks=%d, dynamic=%r aliased=%r  saved %s' % (
                len(mprog.stmts), st, used, mprog.dynamic, mprog.aliased_input, outp))
            print(mprog.source())
            for d in (info or [])[:6]:
                print('      ', d)
        return
    os.makedirs(os.path.join(args.out, 'errors'), exist_ok=True)
    logf = open(os.path.join(args.out, 'log.txt'), 'a', encoding='utf-8')

    def P(*a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    torch._dynamo.config.cache_size_limit = 64
    try:
        import torch._inductor.config as ic
        ic.compile_threads = 1
    except Exception:
        pass
    P('torch', torch.__version__, '| backend', args.backend, '| device', args.device, '| seed', args.seed,
      '| dtype', args.dtype, '| start', time.strftime('%Y-%m-%d %H:%M:%S'))

    stats = {'invalid': 0, 'compile_error': 0, 'value_diff': 0, 'alias_diff': 0, 'pass': 0}
    cases = []
    start = time.time()
    deadline = start + args.minutes * 60
    i = 0
    n_err_saved = 0
    invalid_reasons = {}
    aborted = None
    while time.time() < deadline:
        i += 1
        seed = args.seed * 1000003 + i
        rng = random.Random(seed)
        n = rng.randint(args.min_stmts, args.max_stmts)
        prog = generate(rng, args.dtype, n)
        t0 = time.time()
        st, info = evaluate(prog, args.device, args.backend, seed, args.calls)
        stats[st] += 1
        if st == 'invalid':
            key = str(info)[:100]
            invalid_reasons[key] = invalid_reasons.get(key, 0) + 1
            sticky = args.device == 'cuda' and any(w in str(info) for w in ('CUDA', 'device-side', 'Accelerator'))
            if sticky and not cuda_healthy():
                aborted = 'cuda_context_poisoned'
                with open(os.path.join(args.out, 'errors', 'poisoned_iter_%d.txt' % i), 'w', encoding='utf-8') as f:
                    f.write(prog.source() + '\n# dynamic=%r aliased=%r\n\n%s' % (prog.dynamic, prog.aliased_input, info))
                P('ABORT iter=%d: CUDA context poisoned by an eager-side error; exiting with code 3 so run_budget.py '
                  'can restart in a fresh process: %s' % (i, str(info)[:200]))
                break
        if st == 'compile_error' and n_err_saved < 25:
            n_err_saved += 1
            with open(os.path.join(args.out, 'errors', 'err_%03d.txt' % n_err_saved), 'w', encoding='utf-8') as f:
                f.write(prog.source() + '\n# dynamic=%r aliased=%r\n\n%s' % (prog.dynamic, prog.aliased_input, info))
        if st in ('value_diff', 'alias_diff'):
            case = {'id': len(cases) + 1, 'iter': i, 'seed': seed, 'status': st, 'kinds': prog.kinds(),
                    'dynamic': prog.dynamic, 'aliased': prog.aliased_input, 'n_stmts': n,
                    'inputs': prog.inputs, 'dtype': prog.dtype,
                    'diffs': info[:12], 'source': prog.source()}
            cases.append(case)
            with open(os.path.join(args.out, 'case_%03d.py' % case['id']), 'w', encoding='utf-8') as f:
                f.write(repro_source(prog, args.device, args.backend, seed, args.calls, st))
            with open(os.path.join(args.out, 'case_%03d.json' % case['id']), 'w', encoding='utf-8') as f:
                json.dump(case, f, ensure_ascii=False, indent=1)
            P('[iter %d] %s seed=%d dynamic=%r aliased=%r kinds=%s' % (
                i, st, seed, prog.dynamic, prog.aliased_input, ','.join(prog.kinds())))
            for d in info[:4]:
                P('      ', d)
            if len(cases) >= args.max_cases:
                P('max cases reached')
                break
        if i % 25 == 0:
            P('progress iter=%d %s elapsed=%.0fs last=%.1fs' % (i, stats, time.time() - start, time.time() - t0))
    P('DONE iterations=%d stats=%s cases=%d elapsed=%.0fs' % (i, stats, len(cases), time.time() - start))
    top_invalid = sorted(invalid_reasons.items(), key=lambda kv: -kv[1])[:10]
    for k, v in top_invalid[:5]:
        P('   invalid x%d: %s' % (v, k))

    order = sorted(cases, key=lambda c: (0 if c['status'] == 'value_diff' else 1, c['id']))
    seen = set()
    done = 0
    for c in order:
        if done >= args.minimize or aborted:
            break
        key = (c['status'], tuple(c['kinds']))
        if key in seen:
            continue
        seen.add(key)
        rng = random.Random(c['seed'])
        n = rng.randint(args.min_stmts, args.max_stmts)
        prog = generate(rng, args.dtype, n)
        P('minimizing case %d (%s) ...' % (c['id'], c['status']))
        mprog, used = minimize(prog, args.device, args.backend, c['seed'], args.calls, args.min_budget)
        st, info = evaluate(mprog, args.device, args.backend, c['seed'], args.calls)
        with open(os.path.join(args.out, 'min_case_%03d.py' % c['id']), 'w', encoding='utf-8') as f:
            f.write(repro_source(mprog, args.device, args.backend, c['seed'], args.calls, st))
        P('  -> %d stmts (from %d), status=%s, checks=%d, dynamic=%r aliased=%r' % (
            len(mprog.stmts), len(prog.stmts), st, used, mprog.dynamic, mprog.aliased_input))
        P(mprog.source())
        for d in (info or [])[:6]:
            P('      ', d)
        done += 1
    with open(os.path.join(args.out, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump({'torch': torch.__version__, 'backend': args.backend, 'device': args.device, 'iterations': i,
                   'stats': stats, 'n_cases': len(cases), 'elapsed_s': time.time() - start,
                   'aborted': aborted, 'invalid_reasons': dict(top_invalid)},
                  f, ensure_ascii=False, indent=1)
    P('summary written')
    if aborted:
        sys.exit(3)


if __name__ == '__main__':
    main()
