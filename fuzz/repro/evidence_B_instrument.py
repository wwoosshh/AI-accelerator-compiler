# Instrument gen_alias_from_base to see which base tensor and ViewMeta sequence AOTAutograd uses
# when regenerating the output view `ta[0]` in the synthetic-base case.
import torch
import torch._dynamo  # avoid circular import when importing aot_autograd internals directly
import torch._functorch._aot_autograd.functional_utils as fu
import torch._functorch._aot_autograd.runtime_wrappers as rw
orig = fu.gen_alias_from_base
def spy(aliased_base_tensor, target_meta_tensor, target_requires_grad, target_view_meta_sequence=None, *, replay_views=True):
    seq = target_view_meta_sequence.sequence if target_view_meta_sequence is not None else None
    print(f"  gen_alias_from_base: base shape={tuple(aliased_base_tensor.shape)} offset={aliased_base_tensor.storage_offset()} | "
          f"target shape={tuple(target_meta_tensor.shape)} offset={target_meta_tensor.storage_offset()} | view metas={[type(m).__name__ for m in seq] if seq else None}")
    return orig(aliased_base_tensor, target_meta_tensor, target_requires_grad, target_view_meta_sequence, replay_views=replay_views)
fu.gen_alias_from_base = spy; rw.gen_alias_from_base = spy
def fn(t0, ta):
    t0.add_(100)
    return ta[0]
base = torch.arange(24, dtype=torch.float64).reshape(4, 6)
c0 = base.clone()
out = torch.compile(fn, backend="aot_eager")(c0, c0[1:])
print("  result offset:", out.storage_offset(), "(expected 6)")
