# 기존 PyTorch 이슈에 덧붙일 댓글 (영문)

## pytorch/pytorch#197893 (`x + 0` / `x * 1` returns the input tensor itself)

```
Scatter variant of the same class of problem, reproduced on 2.14.0+cu130 and 2.15.0.dev20260921+cu130 (Inductor, CUDA; `aot_eager` is correct):

    def fn(x):
        y = torch.diagonal_scatter(x, x.diagonal())   # value-wise a no-op, a fresh tensor in eager
        x.fill_(2)
        return y                                       # inductor returns all 2s

Same for torch.select_scatter(x, x.select(0, 1), 0, 1) and torch.slice_scatter(x, x[1:3], 0, 1, 3).
Here the alias does not come from remove_noop_ops but from the reinplacing pass writing the scatter
result into the input buffer (reinplace_inplaceable_ops_core.can_inplace); the later fill_ then
overwrites it. Also `y = x.neg(); ...; y.copy_(x); return y` makes the returned y alias x.
PR with a fix and a regression test in test/inductor/test_inplacing_pass.py: <PR link>
```

## pytorch/pytorch#195451 (Direct generalized_scatter copy-back changes returned output aliasing)

```
The same path also produces wrong *values* when the later mutation of the input is unrelated to the scatter result:

    def fn(x, src):
        y = torch.slice_scatter(x, src[1:3], dim=0, start=1, end=3)
        x.zero_()
        return y      # eager: the scattered values; inductor: all zeros

(`select_scatter` + `fill_` behaves the same; 2.14.0 and 2.15.0.dev20260921, CUDA.) can_inplace only
checks that *some* copy_ into the input exists, not that it writes this node's result back, and does not
check whether the result is observed after that copy_. PR: <PR link>
```
