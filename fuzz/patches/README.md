# 수정 패치 모음 (PyTorch main dc0133e, 2026-09-22 기준)

PyTorch 클론(`C:\Users\s0105\src\pytorch`, 브랜치 `fix-reinplace-input-overwritten-later`)에 3개의 로컬 커밋으로 기록했고, 같은 내용을 여기 `.patch` 파일로 내보냈습니다. 아무 것도 pytorch/pytorch 에 푸시하지 않았습니다.

| 패치 | 고치는 버그 | 상태 |
|---|---|---|
| `0001-inductor-do-not-reinplace-into-input-overwritten-later.patch` | H1, H2, H3 (이슈 #5, #6, #7) | **검증됨**: 재현 스크립트 전부 통과, 새 회귀 테스트가 패치 전 2.14에서 실패·패치 후 통과, 기존 `test_inplacing_pass.py` 회귀 없음(기존 실패 2건은 패치 전에도 실패), 정상 제자리 대입 패턴은 추가 할당 없음 |
| `0002-aot_autograd-synthetic-base-output-alias-regenerate-from-metadata.patch` | B (이슈 #2) | **검증됨**: 모든 변형(크래시 변형 포함) 통과, 새 회귀 테스트 패치 전 실패·후 통과. `test_aotdispatch.py -k "alias or synthetic or mutation"`: 279 통과, 13 실패는 패치 없이도 동일하게 실패(Windows/Inductor CPU 환경 문제), 회귀 없음 |
| `0003-functionalization-unfold-inverse-as_strided_scatter-PROPOSED-UNCOMPILED.patch` | A (이슈 #1) | **제안 단계**: C++ 라 이 PC(MSVC 없음)에서 컴파일·실행 못 함. 수정 공식만 Python 으로 613 케이스 검증(`validate_unfold_inverse_formula.py`, 불일치 0; 현행 `unfold_backward` 는 550 불일치) |

## 각 패치의 요지

### 0001 (Inductor reinplace)
`reinplace_inplaceable_ops_core.can_inplace` 의 입력(placeholder) 분기는 "입력에 되쓰는 `copy_` 가 존재하고, 이 연산 이후 입력의 뷰 사용이 없으면" 결과를 입력 버퍼에 제자리 갱신했다. 그러나 그 `copy_` 가 **다른 값**(예: `x.zero_()` 의 0)을 되쓰는 경우, 입력 버퍼를 공유하게 된 결과가 그래프 출력 등으로 이후에 관측되면 덮어쓰인 값을 보게 된다. 수정: `copy_` 가 이 노드의 결과(또는 그 뷰)를 되쓰는 경우가 아니면서 결과가 `copy_` 이후에도 관측되면 reinplace 하지 않음.

### 0002 (AOTAutograd synthetic base)
`create_synthetic_base_metadata` 가 병합된 입력을 별칭하는 출력의 `base_idx` 를 synthetic base 로 바꾸면서 원래 입력 기준의 `view_meta_sequence` 를 그대로 두었다. 수정: 병합된 경우 시퀀스를 버려 `gen_alias_from_base` 가 출력 자체의 size/stride/storage_offset 으로 `as_strided` 폴백을 쓰게 함(같은 저장소 기준이라 정확). 더 정교한 대안은 입력 재생성 뷰(as_strided)의 ViewMeta 를 시퀀스 앞에 붙이는 것.

### 0003 (functionalization unfold, 제안)
`FunctionalInverses::unfold_inverse` 가 `unfold_backward`(미분 공식, 창 밖 0)를 역변환으로 사용. 수정: `size <= step` 이면 unfold 뷰는 일반 strided 뷰이므로 `base.as_strided_scatter_symint(mutated_view, view_sizes, view_strides, base_offset)` 로 되쓰기. 0-dim 은 기존 동작 유지.

## 적용·테스트 방법

```bash
cd pytorch && git apply /path/to/0001-*.patch /path/to/0002-*.patch
python -m pytest test/inductor/test_inplacing_pass.py -k test_dont_reinplace_into_input_overwritten_later -q
python -m pytest test/functorch/test_aotdispatch.py -k test_input_mutation_aliases_offset_input_output_alias -q
```

설치된 휠에 직접 적용해 확인하려면 `reinplace.py`, `input_output_analysis.py` 를 site-packages 의 같은 경로에 덮어쓰면 된다(원본 백업: `*.orig_nightly`). Windows 에서 PyTorch 자체 테스트 러너(`python test_x.py`)는 출력이 없으니 `python -m pytest` 를 쓴다.

## 남은 버그와 다음 단계

- G (동적 형상 별칭 stride, 이슈 #3): `gen_alias_from_base` 폴백 또는 functionalization 이 만든 갱신 base 의 stride 사용 문제. `runtime_wrappers.py` 의 입력 변이 write-back 과 별칭 재생성 순서를 봐야 함.
- I (dtype 뷰 왕복 순서 위반, 이슈 #8): Inductor 스케줄러의 버퍼 인라인·변이 의존성. `torch/_inductor/scheduler.py` 의 `MutationLayout`/`buffer realize` 조건 조사 필요.
- F (Dynamo 크래시, 이슈 #4): `torch/_dynamo/variables/tensor.py` 에서 `contiguous()`/`to()` 가 자기 자신을 반환하는 경우를 입력 별칭으로 모델링해야 함.
- C (dtype 뷰 별칭 미보존, 이슈 #10), D (나이틀리에서 수정됨, 이슈 #9).
- 업스트림: A/B/H 이슈를 pytorch/pytorch 에 제출한 뒤, 포크에 브랜치를 푸시해 PR(0001, 0002)로 연결. 0003 은 CI 에서 컴파일 검증을 받아야 함.
