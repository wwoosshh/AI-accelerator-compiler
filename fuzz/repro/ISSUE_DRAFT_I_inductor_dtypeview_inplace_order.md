# [inductor] `y = x.view(torch.int32) * 2; y.add_(4); x[:, 2:5] = 2; return y.view(torch.int64)` computes `y` from the *mutated* `x`: dtype-view round trip + in-place op on the intermediate breaks read/write ordering

> 조용한 오답, Inductor 전용. 안정판 2.14.0 과 nightly 2.15.0.dev20260921 (둘 다 CUDA) 에서 재현. aot_eager 정상. 제출 시 `collect_env` 첨부.

### 🐛 Describe the bug

`y` is a fresh tensor derived from a dtype view of the input `x`. It is then updated in place, `x` is mutated
afterwards, and `y` is returned through a dtype view back to `x`'s dtype. Inductor returns values computed
from `x` **after** the mutation, i.e. the read of `x` that defines `y` is evaluated after `x[:, 2:5] = 2`.

```python
import torch

def fn(x):
    y = x.view(torch.int32) * 2     # reads x (as int32 pairs) -> fresh tensor
    y.sub_(-4)                      # in-place update of the intermediate
    x[:, 2:5] = 2                   # mutate the input afterwards
    return y.view(torch.int64)      # must reflect x BEFORE the mutation

x = torch.arange(-12, 12, dtype=torch.int64, device="cuda").reshape(4, 6)
print(fn(x.clone()).flatten().tolist()[:6])
print(torch.compile(fn, backend="inductor")(x.clone()).flatten().tolist()[:6])
```

```
[12884901868, 12884901870, 12884901872, 12884901874, 12884901876, 12884901878]   <- eager
[12884901868, 12884901870, 17179869192, 17179869192, 17179869192, 12884901878]   <- inductor
```

`17179869192 == (4 << 32) + 8` is exactly `(2*2+4, 2*0+4)` as an int32 pair, i.e. the value obtained from
`x == 2` after the assignment. Positions 2..4 are the mutated columns.

All four ingredients are needed (torch 2.14.0 and nightly, `backend="inductor"`, CUDA):

| variant | result |
|---|---|
| as above | MISMATCH |
| `y.add_(4)` instead of `y.sub_(-4)` | MISMATCH |
| `x.fill_(2)` instead of the slice assignment | MISMATCH |
| mutation through a view (`x.unsqueeze(2)[:, 2:5, :] = 2`) | MISMATCH |
| return `y` without `.view(torch.int64)` | OK |
| no in-place op on `y` (`y = x.view(torch.int32) * 2 + 4`) | OK |
| no dtype view (`y = x * 2; y.sub_(-4); ...`) | OK |
| `backend="aot_eager"` | OK |

So the combination "in-place op on an intermediate that is a pointwise function of a dtype view of an input"
+ "later mutation of that input" + "dtype view of the intermediate as output" makes Inductor re-derive the
output from the input after the mutation (the pointwise definition is inlined into the final view kernel
instead of reading the mutated intermediate buffer, and the ordering edge to the input mutation is lost).

### Versions

- torch 2.14.0+cu130 and torch 2.15.0.dev20260921+cu130 (Windows 11, Python 3.12, CUDA 13.0, RTX 4070 Ti, triton-windows 3.8.0)

제출 시 라벨 제안: `module: inductor`, `module: correctness (silent)`, `oncall: pt2`. 관련: #163286 (as_strided lowering throws away .view(dtype), 닫힘), #193760
