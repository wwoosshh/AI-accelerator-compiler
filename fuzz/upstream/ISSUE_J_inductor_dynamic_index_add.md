### 🐛 Describe the bug

With `dynamic=True`, an `index_add_` through a flat `reshape`/`view` alias of an input is dropped entirely when the input was assigned to earlier in the same function. Static shapes and `backend="aot_eager"` (static and dynamic) are correct.

```python
import torch

def fn(x, src):
    v = x.reshape((24,))            # flat alias of x (x.view((24,)) behaves the same)
    x[:1, :8] = 5                   # assign into the base first
    v.index_add_(0, torch.tensor([5], device=x.device), src[:1])   # then accumulate through the alias
    return x

x = torch.arange(-12, 12, dtype=torch.int64, device="cuda").reshape(3, 8)
src = torch.full((2,), 100, dtype=torch.int64, device="cuda")
print(fn(x.clone(), src)[0, 5].item())                                                  # eager   : 105
print(torch.compile(fn, backend="inductor", dynamic=True)(x.clone(), src)[0, 5].item())  # inductor: 5
```

| body | inductor, dynamic=True |
|---|---|
| as above | MISMATCH |
| `x.view(x.shape).view((24,))` or two `reshape`s | MISMATCH |
| return `v` instead of `x` | MISMATCH |
| drop the assignment `x[:1, :8] = 5` | OK |
| `v[5:6].add_(src[:1])` or `v[5] = 7` instead of `index_add_` | OK |
| `dynamic=None`, or `backend="aot_eager"` | OK |

### Generated code

Only two kernels are emitted: one computing the assignment into a temporary and one copying it back into the input. No kernel receives `src` (`arg4_1` is unused) and no kernel compares an index against 5, so the `index_add_` is lost during lowering rather than mis-ordered:

```python
def call(self, args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args        # s70, s68, x, s28, src
    buf0 = empty_strided_cuda((s70, s68), (s68, 1), torch.int64)
    triton_poi_fused_fill_0.run(arg2_1, buf0, s68, s68*s70, ...)
    triton_poi_fused_copy__fill_view_1.run(buf0, arg2_1, s68*s70, ...)
    return (arg2_1,)
```

### Versions

- torch 2.14.0+cu130 and torch 2.15.0.dev20260921+cu130, Windows 11, Python 3.12.10, CUDA 13.0, NVIDIA GeForce RTX 4070 Ti, triton-windows 3.8.0
