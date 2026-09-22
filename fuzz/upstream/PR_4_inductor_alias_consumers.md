## Summary

`GraphLowering.mark_buffer_mutated` realizes the pending users of a mutated buffer so that they read the value before the mutation, but not the users of buffers that merely *alias* it, such as the output of a fallback view kernel (`aten.view.dtype` with a different item size):

```python
y = x.view(torch.int32) * 2   # buf0 = view.dtype(x) aliases x; y is a pending pointwise
y.sub_(-4)
x[:, 2:5] = 2                 # mutation of x: users of x are realized, users of buf0 are not
return y.view(torch.int64)    # y is materialized after the mutation and computed from the mutated x
```

In the scheduler IR the mutation node only depended on the producer of `buf0`, not on its late-realized consumer, and the fill kernel ran first.

## Fix

Index the alias relationships of realized buffers incrementally (`get_inputs_that_alias_output()`) and, in `mark_buffer_mutated`, realize the users of the mutated buffer and, transitively, of every buffer that aliases it.

## Test plan

- New `test_input_mutation_after_dtype_view_consumer` in `test/inductor/test_torchinductor.py` (`CommonTemplate`, runs on every device). Passes with this change on CUDA; the standalone reproducer mismatches on 2.14.0 and 2.15.0.dev20260921 without it.
- `python -m pytest test/inductor/test_torchinductor.py -k "view_dtype or dtype_view or mutation or inplace or alias or copy_"`: 51 passed; the 7 failures on my Windows machine also fail without the patch.
- The reinplace regression tests and the alias-family fuzzer reproducers remain green with this change.

Fixes #TBD (issue for the dtype-view read-after-write ordering bug)
