# [inductor] `x.index_fill_(...); y = x.clone(); x.clamp_(-4, 4); return y` returns the clamped values: the clone after an `index_fill_` is dropped

> 조용한 오답, Inductor 전용. 안정판 2.14.0 과 nightly 2.15.0.dev20260921 (둘 다 CUDA) 에서 재현. aot_eager 정상. (같은 `index_fill_` 이 관여하는 D 초안의 버그는 nightly 에서 고쳐졌으나 이 증상은 남아 있어 별도 이슈.)

### 🐛 Describe the bug

```python
import torch

def fn(x):
    x.index_fill_(1, torch.tensor([1], device=x.device), 4)
    y = x.clone()            # snapshot of x at this point
    x.clamp_(-4, 4)
    return y                 # must be the unclamped snapshot

x = torch.arange(-12, 12, dtype=torch.int64, device="cuda").reshape(4, 6)
print(fn(x.clone()).flatten().tolist()[:8])
print(torch.compile(fn, backend="inductor")(x.clone()).flatten().tolist()[:8])
```

```
[-12, 4, -10, -9, -8, -7, -6, 4]     <- eager
[ -4, 4,  -4, -4, -4, -4, -4, 4]     <- inductor: y aliases x and sees the clamp_
```

Without the `index_fill_`, or with `x.add_(1)` in its place, the clone is kept and the result is correct, so
the buffer produced by the `index_put` lowering of `index_fill_` is treated as freely reusable although it is
both the value written back into the input and the value the clone must snapshot. `backend="aot_eager"` is
correct. Reproduces with `dynamic=None` and `dynamic=True`.

### Versions

- torch 2.14.0+cu130 and torch 2.15.0.dev20260921+cu130 (Windows 11, Python 3.12, CUDA 13.0, RTX 4070 Ti, triton-windows 3.8.0)

제출 시 라벨 제안: `module: inductor`, `module: correctness (silent)`, `oncall: pt2`. 관련: #197489 (no-op clone 제거, 닫힘), #198031
