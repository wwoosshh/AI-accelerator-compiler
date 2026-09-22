# [aot_autograd] Output that is a view of an aliased input (synthetic base) is regenerated from the wrong tensor: silent wrong values (`ta[0]` returns `t0[0]`) or `AssertionError: incorrect out shape after application of ViewMeta sequence`

> 제출 전 확인: 아래 내용은 이 세션에서 실제로 확인한 사실만 담았습니다. 제출은 사용자가 직접 하며, 제출 시 `python -m torch.utils.collect_env` 출력을 Versions 에 붙이면 됩니다.

### 🐛 Describe the bug

When a compiled function receives two inputs that alias each other with a non-zero storage offset
(here `ta = t0[1:]`), mutates the base input in place, and returns a **view of the aliased input**,
AOTAutograd's synthetic-base path regenerates the output view from the wrong tensor. The ViewMeta
sequence that was recorded relative to `ta` (shape `(3, 6)`, storage offset 6) is replayed on the
synthetic base `t0` (shape `(4, 6)`, offset 0). Depending on the view this is either

* **silently wrong**: `ta[0]` comes back as `t0[0]` (storage offset 0 instead of 6), or
* **a crash**: `ta.view(-1)` fails with
  `AssertionError: incorrect out shape after application of ViewMeta sequence: (24,) (actual) vs (18,) (expected)`
  in `torch/_functorch/_aot_autograd/functional_utils.py::gen_alias_from_base`.

The mutated input itself is written back correctly; only the returned view is wrong. Returning `ta`
itself, or mutating `ta` instead of `t0`, works. Reproduces with `aot_eager` and `inductor`, CPU and
CUDA, static and dynamic shapes, on 2.14.0 and on nightly 2.15.0.dev20260921.

```python
import torch

def fn(t0, ta):          # ta is a view of t0 with storage offset 6
    t0.add_(100)
    return ta[0]         # must be t0[1] after the mutation

base = torch.arange(24, dtype=torch.float64).reshape(4, 6)
e0 = base.clone(); eager = fn(e0, e0[1:])
c0 = base.clone(); compiled = torch.compile(fn, backend="aot_eager")(c0, c0[1:])
print(eager.tolist(), eager.storage_offset())
print(compiled.tolist(), compiled.storage_offset())
```

Output:

```
[106.0, 107.0, 108.0, 109.0, 110.0, 111.0] 6      <- eager: ta[0] == t0[1]
[100.0, 101.0, 102.0, 103.0, 104.0, 105.0] 0      <- compiled: t0[0], offset dropped
```

Crash variant (same root cause):

```python
def fn2(t0, ta):
    t0.add_(100)
    return ta.view(-1)   # 18 elements

torch.compile(fn2, backend="aot_eager")(c0, c0[1:])
# AssertionError: incorrect out shape after application of ViewMeta sequence: (24,) (actual) vs (18,) (expected)
#   File ".../torch/_functorch/_aot_autograd/codegen.py:codegen(synthetic_base_wrapper)", in _synthetic_base_wrapper
#   File ".../torch/_functorch/_aot_autograd/codegen.py:codegen(output_alias_wrapper)", in _alias_fn
#   File ".../torch/_functorch/_aot_autograd/functional_utils.py", line 350, in gen_alias_from_base
```

Variants tested (torch 2.14.0, `aot_eager`):

| function body                          | result   |
|----------------------------------------|----------|
| `t0.add_(100); return ta[0]`           | MISMATCH (returns `t0[0]`, offset 0 instead of 6) |
| `t0.add_(100); return ta.view(-1)`     | AssertionError (24 vs 18) |
| `t0.add_(100); return ta[2::2, 2::3, :]` (3-D input) | MISMATCH (offset 16 instead of 22) |
| `t0.add_(100); return ta`              | OK |
| `ta.add_(100); return ta[0]`           | OK |
| `ta.add_(100); return t0[1]`           | OK |
| `t0.add_(100); return t0[1]`           | OK |
| `t0.add_(100); return ta[0] + 0`       | OK (not an alias) |

A random differential fuzzer that generates alias/view/in-place programs hit this pattern
repeatedly; the crash twin also shows up as `ValueError: Cannot view a tensor with shape ... as a tensor
with shape (24,)`, `RuntimeError: shape '[18]' is invalid for input of size 24` and
`RuntimeError: maximum size for tensor at dimension 0 is 2 but size is 10`, all from replaying the
output's ViewMeta chain on the synthetic base instead of on the aliased input.

### Versions

Reproduced on:

- torch 2.14.0+cu130 (Windows 11, Python 3.12, CUDA 13.0, NVIDIA GeForce RTX 4070 Ti), backends `aot_eager` (cpu, cuda) and `inductor` (cuda), `dynamic=None` and `dynamic=True`
- torch 2.15.0.dev20260921+cpu (Windows 11, Python 3.12), backend `aot_eager`

(`python -m torch.utils.collect_env` 출력은 제출 시 첨부)

cc @bdhirsh (aot_autograd) — 제출 시 라벨 제안: `module: aotdispatch`, `module: correctness (silent)`, `oncall: pt2`
