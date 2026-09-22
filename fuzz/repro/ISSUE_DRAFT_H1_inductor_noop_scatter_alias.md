# [inductor] No-op `slice_scatter` / `select_scatter` / `diagonal_scatter(x, x.<same view>)` returns `x` itself, so a later in-place mutation of `x` silently corrupts the functional result (scatter variant of #197893)

> 조용한 오답, Inductor 전용. 이 세션에서 안정판 2.14.0 + CUDA 로 확인(aot_eager 는 정상). nightly + CUDA 재확인은 별도 venv 설치 후 진행 예정. 제출 시 `collect_env` 첨부. #197893 의 스캐터 변형이므로 새 이슈 대신 #197893 에 댓글로 덧붙이는 선택도 가능.

### 🐛 Describe the bug

When the source of a `*_scatter` op is the very view of `self` that it scatters into, the op is a
semantic no-op *in value*, but eager still returns a **new** tensor. Inductor replaces it with `self`
(`remove_noop_ops`), so the returned tensor aliases the graph input, and an in-place mutation of the input
that follows in the same function overwrites the "functional" result.

```python
import torch

def fn(x):
    y = torch.diagonal_scatter(x, x.diagonal())   # value-wise a no-op, but a fresh tensor in eager
    x.fill_(2)
    return y                                       # must still hold the old values

x = torch.arange(-12, 12, dtype=torch.int64, device="cuda").reshape(4, 6)
print(fn(x.clone()).flatten().tolist()[:6])
print(torch.compile(fn, backend="inductor")(x.clone()).flatten().tolist()[:6])
```

```
[-12, -11, -10, -9, -8, -7]     <- eager
[2, 2, 2, 2, 2, 2]              <- inductor (y is x)
```

Same for `torch.select_scatter(x, x.select(0, 1), 0, 1)` and `torch.slice_scatter(x, x[1:3], 0, 1, 3)`.
`backend="aot_eager"` is correct, so the graph is fine and the no-op elimination in Inductor is
responsible, exactly like the arithmetic no-ops `x + 0` / `x * 1` in #197893. Reproduces with
`dynamic=None` and `dynamic=True`.

### Same family: `y.copy_(x)` makes the returned `y` alias the input `x`

```python
def fn(x):
    y = x.neg()                                   # fresh tensor
    x.index_fill_(1, torch.tensor([2], device=x.device), 2)
    y.copy_(x)                                    # y := x (values), y must stay a separate buffer
    return y

out = torch.compile(fn, backend="inductor")(x)
out.add_(1000)                                    # eager: x unchanged / inductor: x is modified (out aliases x)
```

The values returned are right, but `out` shares storage with the graph input, so any later in-place update
by the caller corrupts `x`. Detected by an alias probe (mutate the output, compare the input).

### Versions

torch 2.14.0+cu130 (Windows 11, Python 3.12, CUDA 13.0, RTX 4070 Ti, triton-windows 3.8.0)

제출 시 라벨 제안: `module: inductor`, `module: correctness (silent)`, `oncall: pt2`. 관련: #197893, #197489 (no-op clone 제거), #195451
