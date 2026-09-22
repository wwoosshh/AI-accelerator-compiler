# 수정 패치 모음 (PyTorch main dc0133e, 2026-09-22 기준)

PyTorch 클론(`C:\Users\s0105\src\pytorch`, 브랜치 `fix-reinplace-input-overwritten-later`)에 6개의 로컬 커밋으로 기록했고, 같은 내용을 여기 `.patch` 파일로 내보냈습니다. 아무 것도 pytorch/pytorch 에 푸시하지 않았습니다. 나이틀리 venv(`pt2nightly_cu`)의 site-packages 에는 0001·0002·0004·0005 가 적용된 상태이며 원본 백업은 `*.orig_nightly` 입니다.

| 패치 | 고치는 버그 | 상태 |
|---|---|---|
| `0001-inductor-do-not-reinplace-into-input-overwritten-later.patch` | H1, H2, H3 (이슈 #5, #6, #7) | **검증됨**: 재현 스크립트 전부 통과, 새 회귀 테스트가 패치 전 2.14에서 실패·패치 후 통과, 기존 `test_inplacing_pass.py` 회귀 없음(기존 실패 2건은 패치 전에도 실패), 정상 제자리 대입 패턴은 추가 할당 없음 |
| `0002-aot_autograd-synthetic-base-output-alias-regenerate-from-metadata.patch` | B (이슈 #2) | **검증됨**: 모든 변형(크래시 변형 포함) 통과, 새 회귀 테스트 패치 전 실패·후 통과. `test_aotdispatch.py -k "alias or synthetic or mutation"`: 279 통과, 13 실패는 패치 없이도 동일하게 실패(Windows/Inductor CPU 환경 문제) |
| `0004-dynamo-keep-self-for-noop-methods-returning-self.patch` | F (이슈 #4) | **검증됨**: 재현 9종 통과, 새 회귀 테스트 `test_inplace_view_on_noop_alias_of_input` 패치 전 IndexError·후 통과, Dynamo 표적 테스트(`test_misc/test_functions/test_input_attr_tracking/test_aot_autograd -k "view or inplace or contiguous or alias or mutation or to_"`) 81 통과, 실패 1건은 패치 없이도 실패 |
| `0005-inductor-realize-consumers-of-aliasing-buffers-before-mutation.patch` | I (이슈 #8) | **검증됨**: 변형 8종·퍼저 사례 통과, 새 회귀 테스트 `test_input_mutation_after_dtype_view_consumer` 통과, `test_torchinductor.py -k "view_dtype or dtype_view or mutation or inplace or alias or copy_"` 51 통과, 실패 7건은 패치 없이도 실패, H 계열 재현 계속 정상 |
| `0006-inductor-noop-keep-copy-of-input-mutated-later.patch` | L (비교실험 E2 에서 2026-09-22 발견, `upstream/ISSUE_L_*.md`) | **검증됨**: 재현 변형 54종(`repro/repro_copy_then_view_indexfill.py`, `repro/investigate_L.py`)과 E2 사례 case_019·case_007 통과, 새 회귀 테스트 `test_input_mutation_copy_of_input_mutated_later` 가 패치 전 나이틀리에서 실패·후 통과. 포크 브랜치 `fix/inductor-noop-copy-of-mutated-input`(f01f746) → 업스트림 이슈 #198131, PR #198132 |
| `0003-functionalization-unfold-inverse-as_strided_scatter-PROPOSED-UNCOMPILED.patch` | A (이슈 #1) | C++ 이라 Windows(MSVC 없음)에서는 검증 불가. 수정 공식은 Python 으로 613 케이스 검증(`validate_unfold_inverse_formula.py`, 불일치 0; 현행 `unfold_backward` 는 550 불일치). Docker(Linux) 컨테이너에서 CPU 전용 PyTorch 를 빌드해 검증 진행(`docker_build_verify_A.sh`, 결과는 아래 "A 빌드 검증" 절) |

## 수정 효과 검증 (패치 적용 빌드에 퍼저 재실행)

| 런 | 프로그램 수 | 수정 전 계열 분포 | 수정 후 계열 분포 |
|---|---|---|---|
| aot_eager CPU 10분 | 6,342 | A 62, B 15, G 1 (78건, 15분 런) | **A 56, B 0** |
| Inductor CUDA 12분 | 1,525 | A 22, B 11, K 11, D 4, H2 8, H3 2, 기타 2 (60건) | **A 16, K 1, H·D·I 0**, 새 발견 J 1, C2 1 |

B, H1~H3, I 계열은 패치 적용 빌드의 재실행에서 나오지 않았다. 남은 것은 A(C++ 미적용)와 새 발견 J, C2 인데, 둘은 패치를 되돌린 nightly 와 패치 없는 2.14 에서도 재현되는 **기존 버그**다.

- J: 동적 형상 + 같은 모양 `reshape` 별칭을 통한 `index_add_` 가 2회차 호출에 미반영 (`ISSUE_DRAFT_J_*.md`, 이슈 #11)
- C2: 중간 텐서의 두 뷰 출력이 동적 형상에서 별칭을 잃음(값은 같음, 이슈 #10 코멘트)

## 각 패치의 요지

### 0001 (Inductor reinplace)
`reinplace_inplaceable_ops_core.can_inplace` 의 입력(placeholder) 분기는 "입력에 되쓰는 `copy_` 가 존재하고, 이 연산 이후 입력의 뷰 사용이 없으면" 결과를 입력 버퍼에 제자리 갱신했다. 그 `copy_` 가 **다른 값**(예: `x.zero_()` 의 0)을 되쓰는 경우, 입력 버퍼를 공유하게 된 결과가 그래프 출력 등으로 이후에 관측되면 덮어쓰인 값을 보게 된다. 수정: `copy_` 가 이 노드의 결과(또는 그 뷰)를 되쓰는 경우가 아니면서 결과가 `copy_` 이후에도 관측되면 reinplace 하지 않음.

### 0002 (AOTAutograd synthetic base)
`create_synthetic_base_metadata` 가 병합된 입력을 별칭하는 출력의 `base_idx` 를 synthetic base 로 바꾸면서 원래 입력 기준의 `view_meta_sequence` 를 그대로 두었다. 수정: 병합된 경우 시퀀스를 버려 `gen_alias_from_base` 가 출력 자체의 size/stride/storage_offset 으로 `as_strided` 폴백을 쓰게 함(같은 저장소 기준이라 정확). 더 정교한 대안은 입력 재생성 뷰(as_strided)의 ViewMeta 를 시퀀스 앞에 붙이는 것.

### 0004 (Dynamo, 자기 자신을 반환하는 무연산)
`TensorVariable.call_method` 의 일반 경로에서 결과의 example_value 가 수신자의 example_value 와 같은 객체이고 메서드가 in-place(`_` 접미)가 아니면 새 VariableTracker 대신 `self` 를 반환. 그러면 `v = x.contiguous(); v.unsqueeze_(2)` 가 입력에 대한 in-place 뷰로 인식되어 기존 graph break 경로를 탄다.

### 0005 (Inductor, 별칭 버퍼의 소비자 실체화)
`GraphLowering.mark_buffer_mutated` 가 변이되는 버퍼의 pending 소비자만 실체화하고, 그 버퍼를 **별칭**하는 실체화된 버퍼(`aten.view.dtype` 폴백 출력 등)의 소비자는 놓쳤다. 실체화된 버퍼들의 `get_inputs_that_alias_output()` 을 증분 인덱싱해 별칭을 따라가며 소비자를 실체화.

### 0006 (Inductor, 변이되는 입력의 복사를 noop 으로 지우지 않기)
`remove_noop_ops` 는 `aten.copy(dst, alias(src))`·`aten.clone` 을 원본 `src` 로 치환한다. `src` 가 그래프 안에서 변이되는 입력이고 그 되쓰기 `copy_(src, ·)` 가 치환된 값을 읽는 `copy_(dst, src)` 보다 앞에 놓이면, `dst` 는 변이 **후**의 `src` 를 받는다(`dst[0:, :] = src[0:, :]; src.add_(1)` 이 갱신된 src 를 dst 에 복사). 수정: 입력 저장소별로 그래프 안 첫 `copy_` 변이 위치를 기록하고, 뷰가 아닌 noop 의 원본이 그런 입력을 별칭하며 노드의 사용자 중 하나가 그 변이 뒤에 있으면 치환하지 않는다. 뷰 noop(alias·전체 slice·view)과 사용자가 모두 변이 앞에 있는 복사는 그대로 제거되므로 추가 복사는 정확성에 필요한 곳에만 남는다. 버그는 Dynamo 가 어느 입력을 먼저 올리느냐(첫 사용 순서, 테스트에서는 `canonicalize_output_graph_node_order` 로 이름순)에 따라 나타나서, 회귀 테스트는 원본을 `a`, 대상을 `b` 로 이름 붙여 두 규칙 모두에서 원본이 첫 입력이 되게 했다.

### 0003 (functionalization unfold, C++)
`FunctionalInverses::unfold_inverse` 가 `unfold_backward`(미분 공식, 창 밖 0)를 역변환으로 사용. 수정: `size <= step` 이면 unfold 뷰는 일반 strided 뷰이므로 `base.as_strided_scatter_symint(mutated_view, view_sizes, view_strides, base_offset)` 로 되쓰기. 0-dim 은 기존 동작 유지.

## A 빌드 검증 (Docker)

`docker_build_verify_A.sh` 가 `python:3.12-bookworm` 컨테이너에서 PyTorch main 을 얕게 클론해 0003 을 적용하고 CPU 전용(`USE_CUDA=0`, MKLDNN/분산/테스트 비활성)으로 빌드한 뒤 `repro_unfold_zero.py`, `evidence_A_functionalize.py`, 공식 검증을 실행한다.

**2026-09-22 결과: 미완.** 패치 적용(`PATCH_APPLIED`)과 빌드 시작까지는 성공했으나 `MAX_JOBS=12` 로 15.5GB VM 에서 컴파일하던 중(1830 오브젝트 중 860 지점, `RegisterCPU_*.cpp` 대형 번역 단위) Docker Desktop 의 Linux VM 이 멈췄고, Docker Desktop 을 두 번 재시작해도 엔진(`docker-desktop` WSL 배포판)이 Stopped 상태로 복구되지 않았다. 재시도용 스크립트 `docker_build_verify_A_resume.sh` 는 `MAX_JOBS=6`, `--memory=13g` 로 조정되어 있고 클론이 남아 있으면 재사용한다.

재실행 절차(Docker 가 정상일 때, 약 1~2시간):

```bash
MSYS_NO_PATHCONV=1 docker run -d --name pt-build-A2 --memory=13g -v "C:/Users/s0105/src/ptA:/work" -w /work python:3.12-bookworm bash /work/patches/docker_build_verify_A_resume.sh
docker logs -f pt-build-A2     # BUILD_EXIT=0, IMPORT_OK, 이어서 repro 의 OK/MISMATCH 와 VERIFY_DONE 확인
```

대안: 포크에 브랜치를 올려 PyTorch CI 가 컴파일·테스트하게 하거나, Linux 머신에서 같은 스크립트를 실행.

## 적용·테스트 방법

```bash
cd pytorch && git apply /path/to/000{1,2,4,5}-*.patch
python -m pytest test/inductor/test_inplacing_pass.py -k test_dont_reinplace_into_input_overwritten_later -q
python -m pytest test/functorch/test_aotdispatch.py -k test_input_mutation_aliases_offset_input_output_alias -q
python -m pytest test/dynamo/test_repros.py -k test_inplace_view_on_noop_alias_of_input -q
python -m pytest test/inductor/test_torchinductor.py -k test_input_mutation_after_dtype_view_consumer -q
cd test && python -m pytest inductor/test_torchinductor.py -k test_input_mutation_copy_of_input_mutated_later -q   # 0006 (test/ 에서 실행해야 설치된 torch 를 씀)
```

설치된 휠에 직접 적용해 확인하려면 `reinplace.py`, `input_output_analysis.py`, `_dynamo/variables/tensor.py`, `_inductor/graph.py`, `_inductor/fx_passes/post_grad.py`(0006, `apply_0006_noop_mutated_input.py <경로>` 로 적용) 를 site-packages 의 같은 경로에 덮어쓰면 된다. Windows 에서 PyTorch 자체 테스트 러너(`python test_x.py`)는 출력이 없으니 `python -m pytest` 를 쓴다.

## 남은 버그와 다음 단계

- G (동적 형상 별칭 stride, 이슈 #3): 원인 확정. `ViewAndMutationMeta.make_runtime_safe` 가 심볼릭 입력을 가진 ViewMeta 를 버려 `gen_alias_from_base` 의 폴백이 쓰이는데, 폴백은 traced 출력의 `_base`(functionalization 이 만든, 전치 레이아웃의 갱신 base)의 stride 를 런타임 입력에 그대로 적용한다. 지역 수정으로는 eager 의 원래 stride 를 복원할 정보가 없어 (a) 심볼릭 ViewMeta 를 런타임 값으로 재생하거나 (b) functionalization 이 변이된 base 의 stride 를 보존하도록 하는 설계 결정이 필요. 업스트림 논의용 분석으로 정리.
- J (이슈 #11, 원인 미확정), C·C2 (이슈 #10, 별칭 전용), D (나이틀리에서 수정됨, 이슈 #9).
- 업스트림: 이슈 A/B/F/H/I/J 를 pytorch/pytorch 에 제출하고, 포크에 브랜치를 푸시해 0001·0002·0004·0005 를 PR 로 연결. 0003 은 컨테이너 빌드 결과에 따라 PR 또는 이슈 첨부.
