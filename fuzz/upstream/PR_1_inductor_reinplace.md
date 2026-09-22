## Summary

`reinplace_inplaceable_ops_core.can_inplace` reinplaces a scatter-like op into a graph input whenever a `copy_` epilogue into that input exists and no view of the input is used after the op. That is unsound when the `copy_` writes something *other* than this op's result back and the op's result is still observed afterwards:

```python
y = torch.slice_scatter(x, src, 0, 1, 3)   # reinplaced into x's buffer
x.zero_()                                  # copy_(x, zeros) epilogue overwrites that buffer
return y                                   # returns zeros
```

The same mechanism made value-wise no-op scatters return the input (`y = torch.diagonal_scatter(x, x.diagonal()); x.fill_(2); return y` returns all 2s) and dropped a clone taken after an `index_fill_` (`x.index_fill_(...); y = x.clone(); x.clamp_(...); return y` returns the clamped values).

## Fix

Only reinplace into a graph input when the `copy_` epilogue writes this node's result (or a view of it) back, or when nothing observes the node's result (or a view of it) after that `copy_`. Legitimate patterns keep reinplacing without extra allocations: `x[1:3] = src; return x` and `y = torch.slice_scatter(x, src, ...); x.copy_(y); return x` generate no full-size buffer, as before.

## Test plan

- New `test_dont_reinplace_into_input_overwritten_later` in `test/inductor/test_inplacing_pass.py` covers the three shapes above. It fails on 2.14.0 (`Mismatched elements: 24 / 24`) and passes with this change.
- `python -m pytest test/inductor/test_inplacing_pass.py`: no new failures (the two pre-existing failures on my machine also fail without the patch).
- A random alias/in-place differential fuzzer that produced 14 of 60 divergences from this family on 2.14.0 produced none in 1,525 programs with the patch applied (nightly 2.15.0.dev20260921 + CUDA).

Related: #197893 (arithmetic no-ops returning the input; a different pass), #195451 (aliasing of the copy-back path; values were correct there).

Fixes #TBD (issue for the reinplace-into-overwritten-input bug)
