## Summary

`FunctionalInverses::unfold_inverse` used `unfold_backward` as the inverse view function. That is the autograd formula: every base element that no window covers becomes 0 (and overlapping windows are summed). An in-place mutation through a gapped unfold view (`step > size`) under functionalization / `torch.compile` therefore zeroed the rest of the base:

```python
v = x.unfold(1, 1, 5)
v.add_(v)          # columns 1-4 of x became 0 under torch.compile / torch.func.functionalize
```

## Fix

For `size <= step` the unfold view is an ordinary strided view of base, so scatter the mutated windows back with `as_strided_scatter(base, mutated_view, view_sizes, view_strides, base_storage_offset)`, which keeps every element that no window covers. The overlapping case (`size > step`) keeps raising the existing internal-overlap error; 0-dim bases keep the previous behavior.

The `as_strided_scatter` formulation was validated against eager in-place semantics on 613 (shape, dim, size, step) combinations in Python (0 mismatches; `unfold_backward` mismatches in 550 of them).

## Test plan

- New `test_unfold_with_gaps_inplace_keeps_uncovered_elements` and `test_unfold_single_window_index_fill_keeps_rest` in `test/test_functionalization.py` (`assert_functionalization`, which also covers `TestCrossRefFunctionalization`). Both fail without the change.
- I could not build the C++ change locally (Windows, no MSVC); relying on CI for compilation and the functionalization test suite.

Fixes #TBD (issue for the gapped-unfold functionalization inverse bug)
