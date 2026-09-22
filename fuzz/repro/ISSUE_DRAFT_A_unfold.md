# [functionalization] In-place mutation through a gapped `unfold` view (step > size) silently zeroes every base element not covered by a window under `torch.compile`

> 제출 전 확인: 아래 내용은 이 세션에서 실제로 확인한 사실만 담았습니다. 제출은 사용자가 직접 하며, 제출 시 `python -m torch.utils.collect_env` 출력을 Versions 에 붙이면 됩니다.

### 🐛 Describe the bug

If a tensor is mutated in place *through* an `unfold` view whose windows do **not** cover the whole
base (i.e. `step > size`, so there are gaps between windows), the compiled program overwrites every
base element that no window covers with **0**. Eager leaves those elements untouched.

The bug is in functionalization, not in Inductor: `torch.func.functionalize` alone reproduces it,
and the functionalized graph shows that the mutated view is written back into the base with
`aten.unfold_backward`, which is a *gradient* formula (uncovered positions get 0), not an inverse
view function.

Reproduces with `backend="aot_eager"` and `backend="inductor"`, on CPU and CUDA, with static and
dynamic shapes, on 2.14.0 stable and on the 2.15.0.dev20260921 nightly. The mutation may also be
observed only through the mutated input (no return value), so a caller's tensor is silently corrupted.

```python
import torch

def fn(x):
    v = x.unfold(1, 1, 5)   # windows of size 1 with step 5: columns 0 and 5 only
    v.add_(v)               # doubles x[:, 0] and x[:, 5]; all other columns must stay unchanged
    return x

x = torch.arange(1, 25, dtype=torch.int64).reshape(4, 6)
eager = fn(x.clone())
compiled = torch.compile(fn, backend="aot_eager")(x.clone())
print(eager.tolist())
print(compiled.tolist())
```

Output:

```
eager   : [[2, 2, 3, 4, 5, 12], [14, 8, 9, 10, 11, 24], [26, 14, 15, 16, 17, 36], [38, 20, 21, 22, 23, 48]]
compiled: [[2, 0, 0, 0, 0, 12], [14, 0, 0, 0, 0, 24], [26, 0, 0, 0, 0, 36], [38, 0, 0, 0, 0, 48]]
```

Other shapes of the same bug (all reproduce):

```python
def f_fill(x):                       # single window over a flat view: elements 8..23 become 0
    v = x.view(24).unfold(0, 8, 24)
    v.index_fill_(1, torch.tensor([0, 4], device=x.device), -5)
    return x

def f_setitem(x):                    # rows 1 and 3 (not covered by unfold(0, 1, 2)) become 0
    v = x.unfold(0, 1, 2)
    torch.ops.aten.alias(v)[0] = 7
    return x

def f_mut(x):                        # observed only through the caller's tensor
    v = x.unfold(1, 1, 5)
    v.add_(v)
```

### Root cause (functionalization)

```python
from torch.func import functionalize
from torch.fx.experimental.proxy_tensor import make_fx

out = functionalize(fn)(x.clone())           # same wrong result as torch.compile
print(make_fx(functionalize(fn, remove="mutations_and_views"))(x.clone()).code)
```

```
def forward(self, x_1):
    unfold_copy = torch.ops.aten.unfold_copy.default(x_1, 1, 1, 5)
    add = torch.ops.aten.add.Tensor(unfold_copy, unfold_copy);  unfold_copy = None
    unfold_backward = torch.ops.aten.unfold_backward.default(add, [4, 6], 1, 1, 5);  add = None
    unfold_copy_1 = torch.ops.aten.unfold_copy.default(unfold_backward, 1, 1, 5);  unfold_copy_1 = None
    copy_ = torch.ops.aten.copy_.default(x_1, unfold_backward);  x_1 = copy_ = None
    return unfold_backward
```

`unfold_copy_inverse` (aten/src/ATen/FunctionalInverses.cpp) uses `unfold_backward(mutated_view, base.sizes(), dim, size, step)`
as the inverse of `unfold`. `unfold_backward` is the autograd formula: positions of the base that no
window covers receive 0, and positions covered by several windows receive the *sum*. As a view inverse
it is only correct when the windows tile the base exactly (`step == size` and `size` divides the dim).
For `step > size` the uncovered elements of the base are zeroed (this issue); the overlapping case
(`step < size`) is already rejected with the "internal overlap" error (#165409), and the old
summed-values bug for overlapping windows was #98143.

A correct inverse for the non-overlapping case is to scatter the mutated windows into a copy of the
base, e.g. `as_strided_scatter(base, mutated_view, view.sizes(), view.strides(), view.storage_offset())`,
or equivalently `base.clone()` followed by writing the windows back, so that uncovered positions keep
their base values.

### Versions

Reproduced on:

- torch 2.14.0+cu130 (Windows 11, Python 3.12, CUDA 13.0, NVIDIA GeForce RTX 4070 Ti), backends `aot_eager` (cpu, cuda) and `inductor` (cuda), `dynamic=None` and `dynamic=True`
- torch 2.15.0.dev20260921+cpu (Windows 11, Python 3.12), backend `aot_eager`

(`python -m torch.utils.collect_env` 출력은 제출 시 첨부)

cc @bdhirsh @zou3519 (functionalization) — 제출 시 라벨 제안: `module: functionalization`, `module: correctness (silent)`, `oncall: pt2`
