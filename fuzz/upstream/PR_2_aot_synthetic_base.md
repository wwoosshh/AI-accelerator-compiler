## Summary

When an input that an output aliases is merged into a synthetic base, `create_synthetic_base_metadata` points `base_idx` at the synthetic base but keeps the `view_meta_sequence` recorded relative to the original input. Replaying it on the base drops that input's storage offset / shape:

```python
def fn(t0, ta):        # ta = t0[1:]
    t0.add_(100)
    return ta[0]       # compiled returned t0[0] (offset 0 instead of 6)
```

and `return ta.view(-1)` asserted with `incorrect out shape after application of ViewMeta sequence: (24,) (actual) vs (18,) (expected)` in `gen_alias_from_base`.

## Fix

Drop the ViewMeta sequence for outputs whose aliased input was merged into a synthetic base, so `gen_alias_from_base` falls back to `as_strided` from the output's own size/stride/storage_offset, which are expressed relative to the shared storage. (A more precise alternative would prepend the ViewMeta of the input-regeneration `as_strided` to the sequence; happy to switch if reviewers prefer that.)

## Test plan

- New `test_input_mutation_aliases_offset_input_output_alias` in `test/functorch/test_aotdispatch.py` (`ta[0]` and `ta.view(-1)` with `requires_grad` False/True). It fails on 2.14.0 with the assertion above and passes with this change.
- `python -m pytest test/functorch/test_aotdispatch.py -k "alias or synthetic or mutation"`: 279 passed; the 13 failures on my Windows machine also fail without the patch (Inductor CPU backend needs MSVC there).
- A random alias/in-place differential fuzzer that produced 15 of 78 divergences from this family on 2.14.0 produced none in 6,342 programs with the patch applied.

Fixes #TBD (issue for the synthetic-base output alias bug)
