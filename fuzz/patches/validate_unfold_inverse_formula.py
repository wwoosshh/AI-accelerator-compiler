# Validate the proposed C++ fix for FunctionalInverses::unfold_inverse in Python:
# for non-overlapping windows (size <= step) the unfold view is a plain strided view of base,
# so `base.as_strided_scatter(mutated_view, view_sizes, view_strides, base_offset)` is the exact inverse.
import torch, itertools, random
def unfold_view_params(base, dim, size, step):
    dim = dim % base.dim() if base.dim() > 0 else 0
    sizes, strides = list(base.shape), list(base.stride())
    n = sizes[dim]
    sizes[dim] = (n - size) // step + 1
    sizes.append(size)
    strides[dim] = strides[dim] * step
    strides.append(base.stride(dim))
    return sizes, strides, base.storage_offset()
def proposed_inverse(base, mutated_view, dim, size, step):
    sizes, strides, off = unfold_view_params(base, dim, size, step)
    return base.as_strided_scatter(mutated_view, sizes, strides, off)
def current_inverse(base, mutated_view, dim, size, step):
    return torch.ops.aten.unfold_backward(mutated_view, base.shape, dim, size, step)
random.seed(0); bad_prop = bad_cur = total = 0
for shape in [(24,), (4, 6), (6, 4), (2, 3, 4), (3, 8), (5, 7)]:
    for dim in range(len(shape)):
        n = shape[dim]
        for size in range(1, n + 1):
            for step in range(size, n + 1):         # non-overlapping only (size <= step)
                base = torch.randint(-9, 9, shape, dtype=torch.int64)
                # eager reference: mutate through the real view
                ref = base.clone(); v = ref.unfold(dim, size, step); v.mul_(3); v.add_(1)
                mutated_view = base.unfold(dim, size, step) * 3 + 1        # functional value of the mutated view
                # also try a non-contiguous base (transposed) when 2-D
                for b in ([base, base.t().contiguous().t()] if len(shape) == 2 else [base]):
                    ref_b = b.clone(); vb = ref_b.unfold(dim, size, step); vb.mul_(3); vb.add_(1)
                    mv = b.unfold(dim, size, step) * 3 + 1
                    total += 1
                    bad_prop += not torch.equal(proposed_inverse(b, mv, dim, size, step), ref_b)
                    bad_cur += not torch.equal(current_inverse(b, mv, dim, size, step), ref_b)
print(f"cases={total}  proposed as_strided_scatter mismatches={bad_prop}  current unfold_backward mismatches={bad_cur}")
