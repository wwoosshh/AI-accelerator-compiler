# [inductor][dynamic shapes] In-place `index_add_` through a same-shape `reshape` alias of an input is lost on the *second* call (`dynamic=True`, another pair of aliased inputs present)

> 신규 후보 J. 회귀 퍼징(수정 패치 적용 후) 중 발견. 패치가 없는 2.14.0 과 B 패치를 되돌린 nightly 에서도 동일하게 재현되므로 기존 버그. 아직 최소화된 재현이 6문장이라 추가 축소·원인 분석 필요.

### 🐛 Describe the bug

```python
def fn(t0, t1, t2, ta, k, lst, dct):      # ta = t0.view(-1)  (aliased input pair, otherwise unused)
    v1 = t2.reshape(t2.shape)
    v2 = v1.reshape((24,))
    t2[:1, :8] = 5
    idx4 = torch.tensor([5], device=v2.device)
    v2.index_add_(0, idx4, ta.narrow(0, 0, 1))   # must update t2 through the reshape alias
    return (t2, lst, dct)
```

`torch.compile(fn, backend="inductor", dynamic=True)`: the first call is correct, the **second** call
(different data, different Python int `k`) returns `t2` without the `index_add_` update at flat index 5
(eager: `5 + ta[0]`, compiled: `5`). Observed on torch 2.14.0+cu130 and 2.15.0.dev20260921+cu130 (CUDA).

재현 파일: `fuzz/results/V_ind_cuda_patched/case_011.min.py` (독립 실행: eager 와 컴파일을 2회 호출해 비교).

### 관찰

- `dynamic=None` 으로 바꾸면 최소화 과정에서 사라졌으므로 동적 형상 전용으로 보임.
- 동적 형상에서는 `reshape(t2.shape)` 가 심볼릭 `view` 로 추적되고, 두 번째 호출에서 가드/재컴파일 또는 별칭 처리가 첫 호출과 달라지는 것으로 추정. G(#3) 와 같은 "동적 형상 + 같은 모양 reshape 별칭" 조합.
- 아직 원인 미확정. 다음 단계: `TORCH_LOGS=recompiles,aot_graphs` 로 1회차/2회차 그래프 비교.

### Versions

torch 2.14.0+cu130, torch 2.15.0.dev20260921+cu130 (Windows 11, Python 3.12, RTX 4070 Ti, triton-windows 3.8.0)
