# [inductor] `dst[:] = src[:]; src.add_(1)` copies the *updated* `src` into `dst` (remove_noop_ops replaces a copy of an input by the input itself, which is mutated before the copy's user runs)

### 🐛 Describe the bug

When a function copies one graph input into another through a full-slice view and then mutates the source in place, `torch.compile(backend="inductor")` performs the copy **after** the mutation, so the destination silently receives the post-mutation values. Eager and `aot_eager` are correct.

```python
import torch

def fn(dst, src):
    dst[0:, :] = src[0:, :]   # copy src -> dst (full-slice view as the source)
    src.add_(1)               # then mutate src
    return src

for backend in ("aot_eager", "inductor"):
    torch._dynamo.reset()
    dst_e = torch.zeros(3, 4, device="cuda"); src_e = torch.arange(12., device="cuda").view(3, 4)
    dst_c, src_c = dst_e.clone(), src_e.clone()
    fn(dst_e, src_e)
    torch.compile(fn, backend=backend)(dst_c, src_c)
    print(backend, "dst equal:", torch.equal(dst_c, dst_e), "| eager dst[0]:", dst_e[0].tolist(), "compiled dst[0]:", dst_c[0].tolist())
```

Output (2.14.0+cu130 and nightly 2.15.0.dev20260921+cu130):

```
aot_eager dst equal: True  | eager dst[0]: [0.0, 1.0, 2.0, 3.0] compiled dst[0]: [0.0, 1.0, 2.0, 3.0]
inductor  dst equal: False | eager dst[0]: [0.0, 1.0, 2.0, 3.0] compiled dst[0]: [1.0, 2.0, 3.0, 4.0]
```

Variants checked (all wrong on Inductor, static and dynamic shapes, int64 and float32): the mutation may be `add_`, `mul_`, `index_fill_` (directly or through `src.reshape(-1)` / `src.view(-1)`), the copy may be `dst[:] = src[:, :]`, `dst[0:, :] = src[0:, :]`, or `dst[0:, :] = src[0:, :].clone()`; returning `src`, `dst`, `dst.sum()` or nothing does not matter. Forms whose source is **not** a full-slice view are correct (`dst[0:, :] = src`, `dst.copy_(src[0:, :])`, `dst[1:, :] = src[1:, :]`), and so is any form in which `dst` happens to be lifted as the first graph input (see below for why).

Found by a differential fuzzer for alias/in-place semantics (aliasfuzz); the original program mutated the source through `reshape` + `index_fill_`.

### Root cause

The AOT graph is a correct SSA program (the copy is a fresh value):

```
alias   = aten.alias.default(arg0_1)          # arg0_1 = src (lifted first because src[0:, :] is evaluated first)
copy    = aten.copy.default(arg1_1, alias)    # dst's new value = snapshot of src
add     = aten.add.Tensor(arg0_1, 1)
copy_   = aten.copy_.default(arg0_1, add)     # write back src
copy__1 = aten.copy_.default(arg1_1, copy)    # write back dst
```

`torch/_inductor/fx_passes/post_grad.py::remove_noop_ops` treats `aten.alias` and `aten.copy` (`nop_arg=1`) as noops and replaces both by `arg0_1`. The dst write-back becomes `copy_(arg1_1, arg0_1)`, and it is ordered **after** `copy_(arg0_1, add)` (in-place `add_` after reinplacing), so it reads the mutated `src`. The generated wrapper shows the order:

```
triton_poi_fused_add_0.run(arg0_1, ...)        # src.add_(1) in place
triton_poi_fused_copy__1.run(arg0_1, arg1_1)   # dst <- src, too late
```

The pass only guards against introducing new input/output *aliasing*; it does not consider that the source it substitutes may be **mutated later in the graph** while users of the removed node still run after that mutation. When `dst` is lifted first, the write-backs are emitted in the other order (`copy_(dst, ·)` before `copy_(src, ·)`) and the result is correct by luck, which is why `dst[0:, :] = src` passes.

### Proposed fix

In `remove_noop_ops`, record the earliest in-graph mutation (`aten.copy_`) of each input storage; do not replace a non-view noop (`aten.copy`, `aten.clone`) whose source aliases such an input when any user of the node is ordered after that mutation. View noops (`alias`, full `slice`, `view`) stay eligible, since replacing a view by its base does not change what a later reader observes. With this change all variants above and the fuzzer cases pass; the copy kernel is only kept where correctness requires it. PR: (to be linked).

### Versions

torch 2.14.0+cu130 (Windows 11, RTX 4070 Ti, triton-windows 3.8.0) and 2.15.0.dev20260921+cu130 (same machine). `python collect_env.py` output attached in the linked repository (`fuzz/repro/collect_env_*.txt`).

cc @ezyang @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @chenyang78 @kadeng @muchulee8 @amjames @aakhundov @eellison
