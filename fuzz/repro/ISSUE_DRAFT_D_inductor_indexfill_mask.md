# [inductor] `x.add_(5); x.index_fill_(1, idx, -1); x[x > 0] = 2` silently drops the `index_fill_`: the mutation is scheduled into a dead buffer and the consumer recomputes the pre-mutation value

> **상태: 안정판 2.14.0 에서만 재현, nightly 2.15.0.dev20260921+cu130 에서는 모든 변형이 정상(이미 수정됨).** 따라서 새 이슈로 내기보다 어떤 커밋이 고쳤는지 확인해 2.14.x 릴리스 브랜치 백포트 여부를 묻는 용도로만 쓸 것. 아래는 2.14.0 기준 기록.

### 🐛 Describe the bug

Three chained in-place mutations of a graph input, `x.add_(5)` → `x.index_fill_(1, idx, -1)` → `x[x > 0] = 2`,
produce a wrong result with `backend="inductor"` (CUDA): the effect of `index_fill_` disappears completely.
`backend="aot_eager"` is correct on CPU and CUDA, so the graph handed to Inductor is right and the bug is in
Inductor's scheduling of the mutation.

```python
import torch

def fn(x):
    x.add_(5)
    x.index_fill_(1, torch.tensor([2], device=x.device), -1)   # column 2 := -1
    x[x > 0] = 2                                                # column 2 (-1) must stay -1
    return x

x = torch.arange(1, 13, dtype=torch.int64, device="cuda").reshape(3, 4)
eager = fn(x.clone())
compiled = torch.compile(fn, backend="inductor")(x.clone())
print(eager.tolist())
print(compiled.tolist())
```

Output:

```
eager   : [[2, 2, -1, 2], [2, 2, -1, 2], [2, 2, -1, 2]]
compiled: [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]
```

Also reproduces when the function returns nothing (the caller's `x` is corrupted), with `x += 5` or
`x[:, :] += 5` instead of `x.add_(5)`, and with `dynamic=True`. It does **not** reproduce if the first
mutation is removed, if `x` is a fresh local tensor (`x = x0.clone()`) instead of a graph input, if the
`index_fill_` is replaced by a slice assignment `x[:, 2] = -1`, or if the last statement is `masked_fill_`
or a functional `torch.where`.

### What Inductor generates (TORCH_LOGS="output_code")

```python
def call(self, args):
    arg0_1, = args
    ...
    buf2 = empty_strided_cuda((3, 4), (4, 1), torch.int64)
    # [iadd, gt, setitem_1]: add, gt, where and the copy_ write-back, fused
    triton_poi_fused_add_copy__gt_index_put_lift_fresh_0.run(arg0_1, buf2, arg0_1, 12, stream=raw_stream0)
    # [iadd, tensor, index_fill_]: the index_fill_ (index_put of an expanded scalar)
    triton_poi_fused_add_index_fill_lift_fresh_1.run(buf2, 3, stream=raw_stream0)
    del buf2
    return (arg0_1, )
```

```python
@triton.jit
def triton_poi_fused_add_copy__gt_index_put_lift_fresh_0(in_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK):
    ...
    tmp0 = tl.load(in_ptr0 + (x0), xmask)      # original x
    tmp2 = tmp0 + 5                            # add recomputed inline ...
    tmp4 = tmp2 > 0                            # ... mask taken from the value BEFORE index_fill_
    tmp6 = tl.where(tmp4, 2, tmp2)             # else-value also BEFORE index_fill_
    tl.store(out_ptr1 + (x0), tmp6, xmask)     # buf2
    tl.store(out_ptr2 + (x0), tmp6, xmask)     # arg0_1  (input write-back)

@triton.jit
def triton_poi_fused_add_index_fill_lift_fresh_1(out_ptr0, xnumel, XBLOCK):
    ...
    tl.store(out_ptr0 + (2 + 4*x0), -1, xmask) # writes -1 into column 2 of buf2, which is deleted right after
```

Two things went wrong at once:

1. The `index_put` that implements `index_fill_` (lowered as clone-then-mutate of the `add` result) is
   scheduled **after** its consumer (`gt` / masked `index_put` / `copy_`) and writes into `buf2`, a buffer
   that is deleted immediately and never read.
2. The consumer does not read the mutated buffer at all: the pointwise definition of the buffer (`x + 5`)
   was inlined into the fused kernel, so the mask and the else-value are computed from the value before
   `index_fill_`.

So the mutated buffer's pointwise producer is being inlined into consumers even though a later kernel
mutates that buffer, and the mutation loses its ordering edge. The post-grad graph itself is correct
(`aot_eager` gives the right answer), so the issue is in Inductor's scheduler / mutation tracking for
`index_put` with an expanded (stride-0) value tensor. Possibly related to the fixed #183986 (index ops on
expanded tensors), but that one concerned expanded *self* tensors; here `self` is a normal buffer.

### Note

A different symptom involving the same `index_fill_` lowering (`x.index_fill_(...); y = x.clone(); x.clamp_(...); return y`
returns the clamped values) still reproduces on the nightly and is filed separately (`ISSUE_DRAFT_H3_*`).

### Versions

Reproduced on torch 2.14.0+cu130 (Windows 11, Python 3.12, CUDA 13.0, NVIDIA GeForce RTX 4070 Ti,
triton-windows 3.8.0), `backend="inductor"`, `dynamic=None` and `dynamic=True`.
`backend="aot_eager"` (cpu and cuda) is correct.

(`python -m torch.utils.collect_env` 출력은 제출 시 첨부. nightly + CUDA 재확인 권장.)

제출 시 라벨 제안: `module: inductor`, `module: correctness (silent)`, `oncall: pt2`
