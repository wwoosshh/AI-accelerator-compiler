import torch, torch._dynamo
import torch._functorch._aot_autograd.functional_utils as fu
import torch._functorch._aot_autograd.runtime_wrappers as rw
orig = fu.gen_alias_from_base
def spy(aliased_base_tensor, target_meta_tensor, target_requires_grad, target_view_meta_sequence=None, *, replay_views=True):
    seq = target_view_meta_sequence.sequence if target_view_meta_sequence is not None else None
    b = target_meta_tensor._base
    print(f"  base(input) shape={tuple(aliased_base_tensor.shape)} stride={aliased_base_tensor.stride()} | target shape={tuple(target_meta_tensor.shape)} stride={tuple(target_meta_tensor.stride())} "
          f"| target._base={'None' if b is None else (tuple(b.shape), tuple(b.stride()))} | view metas={[type(m).__name__ for m in seq] if seq else None} replay={replay_views}")
    out = orig(aliased_base_tensor, target_meta_tensor, target_requires_grad, target_view_meta_sequence, replay_views=replay_views)
    print(f"  -> out stride={out.stride()}")
    return out
fu.gen_alias_from_base = spy; rw.gen_alias_from_base = spy
def fn(t0):
    v2 = t0.transpose(1, 2); v3 = t0.view(t0.shape); v2.masked_fill_(v2 > 2, 5); return v3
for dyn in (None, True):
    print(f"dynamic={dyn}")
    torch._dynamo.reset()
    x = torch.arange(-12, 12, dtype=torch.int64).reshape(2, 3, 4)
    torch.compile(fn, backend="aot_eager", dynamic=dyn)(x.clone())
