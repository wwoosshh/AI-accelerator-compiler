# [aot_autograd][dynamic shapes] Output alias `x.view(x.shape)` of an input that was mutated through a transposed view is regenerated with transposed strides: wrong element order (`dynamic=True` only)

> 조용한 오답. 이 세션에서 안정판 2.14.0(aot_eager CPU) 과 nightly 2.15.0.dev20260921 에서 확인. 정적 형상에서는 발생하지 않음. 제출 시 `python -m torch.utils.collect_env` 첨부.

### 🐛 Describe the bug

With `dynamic=True`, if a graph input is mutated in place *through a transposed view* and the function
returns another alias of the same input (`t0.view(t0.shape)` or `t0.reshape(t0.shape)`), the returned
tensor aliases the input storage but has the **strides of the transposed view's layout** instead of the
input's strides. The elements therefore come back permuted (3-D) or plainly wrong (2-D). The mutated input
itself is correct. Static shapes are fine, and so is returning the input itself.

```python
import torch

def fn(t0):
    v2 = t0.transpose(1, 2)
    v3 = t0.view(t0.shape)        # alias of t0 with t0's strides (12, 4, 1)
    v2.masked_fill_(v2 > 2, 5)    # mutate t0 through the transposed view
    return v3

base = torch.arange(-12, 12, dtype=torch.int64).reshape(2, 3, 4)
e0 = base.clone(); eager = fn(e0)
c0 = base.clone(); compiled = torch.compile(fn, backend="aot_eager", dynamic=True)(c0)
print(eager.stride(), eager.flatten().tolist()[:12])
print(compiled.stride(), compiled.flatten().tolist()[:12])
print(torch.equal(e0, c0))   # the mutated input is correct
```

Output:

```
(12, 4, 1) [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1]
(12, 1, 3) [-12, -9, -6, -3, -11, -8, -5, -2, -10, -7, -4, -1]      <- strides of (2,4,3)-contiguous transposed back
True
```

`(12, 1, 3)` is exactly the stride pattern of a `(2, 4, 3)` contiguous tensor transposed on dims 1 and 2,
i.e. the layout of the *updated base* that functionalization produces
(`transpose_copy(masked_fill(transpose_copy(t0)))`). The output alias is regenerated with the strides of
that intermediate instead of the strides of the eager view of the input.

Variants (torch 2.14.0, `backend="aot_eager"`, CPU; identical on nightly):

| body | `dynamic=None` | `dynamic=True` |
|---|---|---|
| `v2 = t0.transpose(1,2); v3 = t0.reshape(t0.shape); v2.masked_fill_(...); return v3` | OK | MISMATCH, strides (12,1,3) |
| same with `t0.view(t0.shape)` | OK | MISMATCH, strides (12,1,3) |
| alias created *after* the mutation | OK | MISMATCH, strides (12,1,3) |
| 2-D: `v2 = t0.t(); v3 = t0.reshape(t0.shape); v2.masked_fill_(...); return v3` | OK | MISMATCH, strides (1,4) vs (6,1), wrong values |
| no mutation | OK | OK |
| mutate `t0` directly (not through the transposed view) | OK | OK |
| `return t0` (the input itself) | OK | OK |

### Versions

- torch 2.14.0+cu130 (Windows 11, Python 3.12), `backend="aot_eager"` (CPU)
- torch 2.15.0.dev20260921+cpu (Windows 11, Python 3.12)

제출 시 라벨 제안: `module: aotdispatch`, `module: dynamic shapes`, `module: correctness (silent)`, `oncall: pt2`
