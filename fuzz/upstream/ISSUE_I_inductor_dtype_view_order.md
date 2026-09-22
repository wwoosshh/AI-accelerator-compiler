### 🐛 Describe the bug

`y` is a fresh tensor derived from a dtype view of the input `x`. It is updated in place, `x` is mutated afterwards, and `y` is returned through a dtype view back to `x`'s dtype. Inductor returns values computed from `x` **after** the mutation, i.e. the read of `x` that defines `y` is evaluated after `x[:, 2:5] = 2`.

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
[12884901868, 12884901870, 12884901872, 12884901874, 12884901876, 12884901878]   # eager
[12884901868, 12884901870, 17179869192, 17179869192, 17179869192, 12884901878]   # inductor
```

`17179869192 == (4 << 32) + 8` is `(2*2+4, 2*0+4)` as an int32 pair, i.e. the value obtained from `x == 2` after the assignment; positions 2..4 are the mutated columns.

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

Reproduces on torch 2.14.0+cu130 and 2.15.0.dev20260921+cu130 (CUDA), `dynamic=None` and `dynamic=True`.

### Root cause

The post-grad graph is correct (`aot_eager` gives the right answer). `aten.view.dtype` with a different item size is lowered as a fallback kernel whose output `buf0` aliases the input `arg0_1`. In the scheduler IR the mutation node (the assignment into `arg0_1`) only depends on the *producer* of `buf0`, not on its later-realized consumer `mul/sub`:

```
op1: SchedulerNode(ComputedBuffer)        # x[:, 2:5] = 2, MutationLayout on arg0_1
op1.unmet_dependencies = [WeakDep(name='buf0', mutating_buf='buf1', is_fake=False)]
op2: SchedulerNode(ComputedBuffer)        # y = view.dtype(x) * 2 - (-4), realized late
op2.unmet_dependencies = [MemoryDep('buf0', c0, {c0: 48})]
```

`GraphLowering.mark_buffer_mutated(name)` realizes the pending users of the mutated buffer so they read the old value, but not the users of buffers that merely *alias* it (the fallback view output). The pointwise `mul/sub` is therefore materialized after the mutation. A PR that follows alias relationships of realized buffers in `mark_buffer_mutated`, plus a regression test in `test/inductor/test_torchinductor.py`, follows.

### Versions

- torch 2.14.0+cu130 and torch 2.15.0.dev20260921+cu130, Windows 11, Python 3.12.10, CUDA 13.0, NVIDIA GeForce RTX 4070 Ti, triton-windows 3.8.0
