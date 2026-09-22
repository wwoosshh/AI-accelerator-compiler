# [inductor] `y = torch.slice_scatter(x, src, ...); x.zero_(); return y` returns zeros: the scatter result is reinplaced into the input buffer that a later mutation overwrites (wrong values, not just aliasing as in #195451)

> 조용한 오답, Inductor 전용. 안정판 2.14.0 + CUDA 확인(aot_eager 정상). nightly + CUDA 재확인 예정. #195451(값은 맞고 별칭만 다름)과 같은 경로로 보이며, 여기서는 값까지 틀림.

### 🐛 Describe the bug

A functional `slice_scatter` / `select_scatter` whose `self` is a graph input is turned into an in-place
update of that input's buffer by Inductor. If the input is then mutated in place later in the same function,
the mutation overwrites the buffer that the returned tensor shares, so the returned value is wrong.

```python
import torch

def fn(x, src):
    y = torch.slice_scatter(x, src[1:3], dim=0, start=1, end=3)   # fresh tensor in eager
    x.zero_()
    return y

x   = torch.arange(24, dtype=torch.int64, device="cuda").reshape(4, 6)
src = torch.arange(-12, 12, dtype=torch.int64, device="cuda").reshape(4, 6)
print(fn(x.clone(), src.clone()).flatten().tolist()[:8])
print(torch.compile(fn, backend="inductor")(x.clone(), src.clone()).flatten().tolist()[:8])
```

```
[0, 1, 2, 3, 4, 5, -6, -5]      <- eager
[0, 0, 0, 0, 0, 0, 0, 0]        <- inductor
```

`select_scatter` behaves the same (`x.fill_(7)` afterwards gives all 7s). `backend="aot_eager"` is
correct. Reproduces with `dynamic=None` and `dynamic=True`. Replacing the scatter with a clone plus slice
assignment (`y = x.clone(); y[1:3] = src[1:3]`) is correct, so it is specific to the scatter lowering /
reinplacing path.

#195451 reports the same path with `x.copy_(updated)` where the values happen to be right and only the alias
contract differs; here the later mutation makes the values wrong.

### Versions

torch 2.14.0+cu130 (Windows 11, Python 3.12, CUDA 13.0, RTX 4070 Ti, triton-windows 3.8.0)

제출 시 라벨 제안: `module: inductor`, `module: correctness (silent)`, `oncall: pt2`. 관련: #195451, #195285
