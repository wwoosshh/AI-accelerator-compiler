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

In `remove_noop_ops`, record the earliest in-graph mutation (`aten.copy_`) of each input storage; do not replace a non-view noop (`aten.copy`, `aten.clone`) whose source aliases such an input when any user of the node is ordered after that mutation. View noops (`alias`, full `slice`, `view`) stay eligible, since replacing a view by its base does not change what a later reader observes. With this change all variants above and the fuzzer cases pass; the copy kernel is only kept where correctness requires it. A fix is proposed in the linked PR.

### Versions

torch 2.14.0+cu130 (Windows 11, RTX 4070 Ti, triton-windows 3.8.0) and nightly 2.15.0.dev20260921+cu130 on the same machine; both reproduce.

<details><summary>collect_env (stable 2.14.0)</summary>

```
PyTorch version: 2.14.0+cu130
Is debug build: False
CUDA used to build PyTorch: 13.0
ROCM used to build PyTorch: N/A

OS: Microsoft Windows 11 Home (10.0.26200 64��Ʈ)
GCC version: Could not collect
Clang version: Could not collect
CMake version: Could not collect
Libc version: N/A

Python version: 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)] (64-bit runtime)
Python platform: Windows-11-10.0.26200-SP0
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: 
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 4070 Ti
Nvidia driver version: Could not collect
cuDNN version: Could not collect
Is XPU available: False
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: False
Caching allocator config: N/A

CPU:
Name: Intel(R) Core(TM) i7-14700K
Manufacturer: GenuineIntel
Family: 198
Architecture: 9
ProcessorType: 3
DeviceID: CPU0
CurrentClockSpeed: 3400
MaxClockSpeed: 3400
L2CacheSize: 28672
L2CacheSpeed: None
Revision: None

Versions of relevant libraries:
[pip3] numpy==2.5.3
[pip3] torch==2.14.0+cu130
[pip3] triton-windows==3.8.0.post28
[conda] Could not collect
```
</details>

<details><summary>collect_env (nightly 2.15.0.dev20260921)</summary>

```
PyTorch version: 2.15.0.dev20260921+cu130
Is debug build: False
CUDA used to build PyTorch: 13.0
ROCm SDK used to build PyTorch: N/A
HIP used to build PyTorch: N/A

OS: Microsoft Windows 11 Home (10.0.26200 64��Ʈ)
GCC version: Could not collect
Clang version: Could not collect
CMake version: Could not collect
Libc version: N/A

Python version: 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)] (64-bit runtime)
Python platform: Windows-11-10.0.26200-SP0
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: 
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 4070 Ti
Nvidia driver version: Could not collect
cuDNN version: Could not collect
Is XPU available: False
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: False
Caching allocator config: N/A

CPU:
Name: Intel(R) Core(TM) i7-14700K
Manufacturer: GenuineIntel
Family: 198
Architecture: 9
ProcessorType: 3
DeviceID: CPU0
CurrentClockSpeed: 3400
MaxClockSpeed: 3400
L2CacheSize: 28672
L2CacheSpeed: None
Revision: None

Versions of relevant libraries:
[pip3] numpy==2.5.3
[pip3] torch==2.15.0.dev20260921+cu130
[pip3] triton-windows==3.8.0.post28
[conda] Could not collect
```
</details>
