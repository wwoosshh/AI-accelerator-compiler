### 🐛 Describe the bug

If a tensor is mutated in place *through* an `unfold` view whose windows do not cover the whole base (`step > size`, i.e. gaps between windows), the compiled program overwrites every base element that no window covers with **0**. Eager leaves those elements untouched. The mutation can also be observed only through the mutated input, so a caller's tensor is silently corrupted.

```python
import torch

def fn(x):
    v = x.unfold(1, 1, 5)   # windows of size 1 with step 5: columns 0 and 5 only
    v.add_(v)               # doubles x[:, 0] and x[:, 5]; all other columns must stay unchanged
    return x

x = torch.arange(1, 25, dtype=torch.int64).reshape(4, 6)
print(fn(x.clone()).tolist())
print(torch.compile(fn, backend="aot_eager")(x.clone()).tolist())
```

```
[[2, 2, 3, 4, 5, 12], [14, 8, 9, 10, 11, 24], [26, 14, 15, 16, 17, 36], [38, 20, 21, 22, 23, 48]]   # eager
[[2, 0, 0, 0, 0, 12], [14, 0, 0, 0, 0, 24], [26, 0, 0, 0, 0, 36], [38, 0, 0, 0, 0, 48]]            # compiled
```

Other shapes of the same bug (all reproduce): `x.view(24).unfold(0, 8, 24)` followed by `index_fill_` zeroes elements 8..23; `x.unfold(0, 1, 2)` followed by a `__setitem__` on `torch.ops.aten.alias(v)` zeroes rows 1 and 3; the same with no return value corrupts the caller's tensor.

Reproduces with `backend="aot_eager"` and `backend="inductor"`, CPU and CUDA, `dynamic=None` and `dynamic=True`, on 2.14.0 and on the 2.15.0.dev20260921 nightly.

### Root cause

`torch.func.functionalize` alone reproduces it, and the functionalized graph writes the mutated view back with `aten.unfold_backward`:

```python
from torch.func import functionalize
from torch.fx.experimental.proxy_tensor import make_fx
print(make_fx(functionalize(fn, remove="mutations_and_views"))(x.clone()).code)
```

```
unfold_copy = torch.ops.aten.unfold_copy.default(x_1, 1, 1, 5)
add = torch.ops.aten.add.Tensor(unfold_copy, unfold_copy)
unfold_backward = torch.ops.aten.unfold_backward.default(add, [4, 6], 1, 1, 5)
copy_ = torch.ops.aten.copy_.default(x_1, unfold_backward)
```

`FunctionalInverses::unfold_inverse` (aten/src/ATen/FunctionalInverses.cpp) uses `unfold_backward(mutated_view, base.sizes(), dim, size, step)` as the inverse of `unfold`. `unfold_backward` is the autograd formula: base positions that no window covers receive 0, positions covered by several windows receive the sum. As a view inverse it is only correct when the windows tile the base exactly. The overlapping case (`step < size`) already raises the "internal overlap" error (#165409), and the old summed-values bug for overlapping windows was #98143; the gapped case (`step > size`) passes the check and silently zeroes the uncovered elements.

For `size <= step` the unfold view is an ordinary strided view of base, so `as_strided_scatter(base, mutated_view, view_sizes, view_strides, base_offset)` is an exact inverse. I validated that formula against eager in-place semantics on 613 (shape, dim, size, step) combinations in Python (0 mismatches; `unfold_backward` mismatches in 550 of them). A PR with this change plus regression tests in `test/test_functionalization.py` follows.

### Versions

Reproduced on:
- torch 2.14.0+cu130, Windows 11, Python 3.12.10, CUDA 13.0, NVIDIA GeForce RTX 4070 Ti (backends `aot_eager` cpu/cuda, `inductor` cuda)
- torch 2.15.0.dev20260921+cu130 and +cpu, same machine

Full `collect_env` output: see attached (collect_env_stable_2.14.0.txt / collect_env_nightly_2.15.0.dev20260921.txt).

cc @bdhirsh @zou3519
