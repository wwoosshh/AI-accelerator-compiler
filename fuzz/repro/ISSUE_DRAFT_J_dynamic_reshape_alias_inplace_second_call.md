# [inductor][dynamic shapes] `index_add_` through a flat `reshape`/`view` alias of an input is dropped when the input was assigned to earlier in the same function (`dynamic=True` only)

> 신규 후보 J. 패치 적용 빌드의 회귀 퍼징에서 발견됐고, 패치 없는 2.14.0 과 B 패치를 되돌린 nightly 에서도 동일하게 재현되므로 기존 버그. Inductor 전용(aot_eager 정상), `dynamic=True` 전용.

### 🐛 Describe the bug

```python
import torch

def fn(x, src):
    v = x.reshape((24,))            # flat alias of x  (x.view((24,)) behaves the same)
    x[:1, :8] = 5                   # assign into the base first
    v.index_add_(0, torch.tensor([5], device=x.device), src[:1])   # then accumulate through the alias
    return x

x = torch.arange(-12, 12, dtype=torch.int64, device="cuda").reshape(3, 8)
src = torch.full((2,), 100, dtype=torch.int64, device="cuda")
print(fn(x.clone(), src)[0, 5].item())                                              # eager   : 105
print(torch.compile(fn, backend="inductor", dynamic=True)(x.clone(), src)[0, 5].item())  # inductor: 5
```

The `index_add_` has no effect at all: the returned `x` (and the mutated input) only reflect the assignment.
Static shapes (`dynamic=None`) and `backend="aot_eager"` (static and dynamic) are correct, so the graph
handed to Inductor is right and the bug is in Inductor's lowering/scheduling under symbolic shapes.

Variants (torch 2.14.0+cu130 and 2.15.0.dev20260921+cu130, CUDA):

| body | inductor, dynamic=True |
|---|---|
| as above | MISMATCH |
| `v = x.reshape(x.shape).reshape((24,))` (two reshapes, as found by the fuzzer) | MISMATCH |
| `x.view(x.shape)` / `.view((24,))` instead of reshape | MISMATCH |
| return `v` instead of `x` | MISMATCH |
| drop the assignment `x[:1, :8] = 5` | OK |
| `v[5:6].add_(src[:1])` instead of `index_add_` | OK |
| `v[5] = 7` instead of `index_add_` | OK |
| any variant with `dynamic=None`, or `backend="aot_eager"` | OK |

### Generated code (TORCH_LOGS=output_code)

Only two kernels are emitted: one computing the assignment into a temporary, one copying it back into
the input. **No kernel receives `src`** (`arg4_1` is unused) and no kernel compares an index against 5:

```python
def call(self, args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args        # s70, s68, x, s28, src
    ...
    buf0 = empty_strided_cuda((s70, s68), (s68, 1), torch.int64)
    triton_poi_fused_fill_0.run(arg2_1, buf0, s68, s68*s70, ...)   # x[:1, :8] = 5 (+ a masked re-load of x itself)
    triton_poi_fused_copy__fill_view_1.run(buf0, arg2_1, s68*s70, ...)  # write back into x
    return (arg2_1,)
```

So under symbolic shapes the `index_add_` (an `index_put`/`scatter_add` into a symbolic-size view of a
buffer that is also mutated by a preceding assignment) is lost during lowering rather than mis-ordered.
Root cause not yet isolated; next step is to diff the post-grad graph against the pre-lowering IR
(`TORCH_COMPILE_DEBUG=1`) for the `index_put` node.

재현 파일: `fuzz/repro/repro_J_variants.py`, `fuzz/repro/dump_code_J.py`, `fuzz/repro/inductor_code_J.txt`, 퍼저 원본 `fuzz/results/V_ind_cuda_patched/case_011.min.py`.

### Versions

torch 2.14.0+cu130, torch 2.15.0.dev20260921+cu130 (Windows 11, Python 3.12, RTX 4070 Ti, triton-windows 3.8.0)

제출 시 라벨 제안: `module: inductor`, `module: dynamic shapes`, `module: correctness (silent)`, `oncall: pt2`
