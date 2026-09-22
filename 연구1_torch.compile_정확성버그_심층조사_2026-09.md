# 연구 1: 딥러닝 컴파일러의 "조용한 오답" 버그 — 심층 조사와 실증 검증

- 작성일: 2026년 9월 22일
- 대상: PyTorch 2 컴파일러 스택(TorchDynamo → AOTAutograd/functionalization → TorchInductor)의 정확성 버그 중 **메모리 계열**(in-place, 뷰/별칭, 입력 변이 write-back, Python 부작용)
- 이 문서의 두 부분: (1) 문헌·이슈 트래커 심층 조사, (2) 이 PC에서 직접 만든 특화 퍼저로 실제 버그가 나오는지 검증한 결과
- 코드와 재현 스크립트: `fuzz/` (사용법은 `fuzz/README.md`)

---

## 0. 결론 요약

1. **버그는 나온다. 빨리 나온다.** 별칭·뷰·in-place 에 특화한 차등 퍼저를 만들어 torch 2.14.0 안정판에서 약 21,000개 프로그램(총 45분)을 돌린 결과, 불일치 사례 151건이 나왔고 이를 분류·최소화해 **나이틀리(2.15.0.dev20260921)에서도 재현되는 신규 조용한 오답 버그 7계열(A, B, G: AOTAutograd/functionalization, H1, H2, H3, I: Inductor)**, **안정판 2.14.0에만 있고 나이틀리에서는 이미 수정된 Inductor 버그 1건(D)**, **별칭 의미 차이 1건(C)**, **크래시형 Dynamo 버그 1군(F)**, **알려진 버그 1건(#197893) 재발견**을 얻었다. 첫 20초(aot_eager)와 75초(Inductor CUDA) 스모크 실행에서 이미 A와 #197893이 나왔다. 모두 2~4문장 재현이며 이슈 초안 9개가 `fuzz/repro/`에 있다.
2. **AOTAutograd/functionalization 계열의 근본 원인을 특정했다.** A는 functionalization이 `unfold`의 역변환으로 미분 공식 `unfold_backward`를 써서 창 밖 원소를 0으로 덮어쓰는 것, B는 synthetic base 경로가 출력 뷰의 ViewMeta 시퀀스를 별칭 입력이 아닌 base에 재적용해 storage offset을 잃는 것(크래시 쌍둥이 동반), G는 동적 형상에서 변이된 입력의 출력 별칭이 전치된 stride로 재생성되는 것이다.
3. **Inductor 계열은 모두 `aot_eager`에서 정상이라 스케줄러·패스 문제로 귀속된다.** H1은 무연산 `*_scatter(x, x.view)`와 `copy_`가 입력 별칭을 반환, H2는 `slice_scatter` 결과를 나중에 변이되는 입력 버퍼에 reinplace, H3는 `index_fill_` 뒤의 `clone()` 제거, I는 dtype 뷰 왕복과 중간 텐서 in-place가 결합되면 입력 변이 이후 값으로 재계산. D(`index_fill_` 변이가 죽은 버퍼에 스케줄)는 나이틀리에서 고쳐졌다.
4. 이 영역은 **경쟁이 매우 치열**하다. 2026년 5월 이후 `module: correctness (silent)` 라벨 이슈가 365건이고, 소수 신고자(상위 4명이 84건)가 자동화 도구로 대량 신고 중이며, 이번 조사 중 하루(9월 21일)에만 10건 이상이 올라왔다. 연구로서의 차별점은 "버그 몇 개"가 아니라 **오라클과 생성기의 설계**(별칭 프로브, 정수값 정확 비교, UB 회피, 입력 변이 write-back 비교)와 **컴포넌트별 근본 원인 분석**에 두어야 한다.

---

## 1. 문헌·이슈 트래커 심층 조사

### 1.1 실증 연구: torch.compile 정확성 버그 116건 (arXiv 2604.08720, 2026년 4월)

- 2023-04 ~ 2025-04 PyTorch 이슈 2,542건 중 확인된 정확성 버그 116건. PyTorch 고우선순위 이슈의 19.2%가 예외·크래시·경고 없이 틀린 출력을 내는 컴파일 버그로, 두 번째로 흔한 범주.
- 분류: 연산자 관련 37.9%(연산자 변환 18.1%, 저수준 코드 생성 19.8%), **메모리 관련 33.6%(in-place 처리 21.6%, 메모리 레이아웃 충돌 12.1%)**, 그래프 관련 19.8%(그래프 의미 포착 15.5%, 그래프 캐싱 4.3%), 기타 8.6%.
- in-place 버그의 유발 패턴: `view`, `expand`, `unfold`, reshape/permute/슬라이싱으로 만든 별칭에 in-place 적용; 커스텀 연산자나 텐서를 담은 list/dict를 통한 암묵적 별칭.
- 근본 원인 위치: Dynamo(커스텀 로직의 불완전한 심볼릭 트레이싱, 상태 변이를 지우는 과도한 최적화), functionalization(in-place 최적화 오류 → 잘못된 그래프), 메모리 관리(별칭 추적 오류, 레이아웃 호환성 검사 누락), Inductor(수치 불안정 코드 생성, 경계 사례 누락).
- 기존 퍼저 5종(NeuRI, WhiteFox, DeepConstr, Opera, NNSmith)을 합쳐도 벤치마크 77건 중 26건(33.8%)만 검출. 그래프 관련 버그는 0건 검출. 원인: 단발 차등 테스트 오라클의 한계(반복 실행이 필요한 캐싱 버그), API 커버리지 부족(비연산 API, 컨텍스트), `unfold`·중첩 인덱싱 같은 복잡한 사용 패턴 부재.
- 제안 도구 AlignGuard는 LLM으로 이슈·커밋에서 패턴을 추출해 기존 테스트를 변이하는 개념증명. 새 버그 23건(고우선순위 14건, 수정 10건). 데이터셋·코드 공개 URL은 논문 본문에서 확인되지 않음.

### 1.2 관련 퍼저와 검증 연구

| 도구 | 접근 | 대상 | 우리와의 차이 |
|---|---|---|---|
| NNSmith (2022) | 문법 기반 모델 생성 | TVM, ORT, TensorRT, PyTorch | 연산자 그래프 중심, in-place·별칭 없음 |
| NeuRI (2023) | 규칙 추론 기반 생성 | 다수 | 연산자 커버리지 중심 |
| TorchProbe (2023, arXiv 2310.20078) | 의미 보존 변환(동적 제어 흐름, 클로저) | torch.compile, Triton | 동적 Python 기능 중심, 20건 발견 |
| WhiteFox (OOPSLA 2024) | LLM이 최적화 소스코드를 읽고 트리거 테스트 생성 | Inductor, TF-XLA, TFLite | 최적화 패스 중심, 101건(92 확정) |
| OATest (2025, arXiv 2511.18918) | 최적화 인지 테스트 생성 | TVM, ORT, TensorRT, OpenVINO | torch.compile 대상 아님 |
| AlignGuard (2026) | LLM 변이 + 차등 오라클 | torch.compile | 개념증명, 23건 |
| Volta (2025, arXiv 2511.12638) | GPU 커널 등가성 검사(건전·완전, 특정 클래스) | 수작업·LLM·컴파일러 생성 커널 | 형식 검증, 텐서 프로그램 수준 아님 |
| 컴파일 불일치 백도어 (IEEE S&P 2026, arXiv 2509.11173) | 컴파일 전후 의미 차이를 공격에 이용 | 상용 DL 컴파일러 3종 | 방어 도구가 열린 문제 |

**공백**: 텐서 **별칭 구조 자체**를 오라클로 삼는 도구가 없다. 기존 오라클은 값만 비교하므로 "값은 같지만 별칭 관계가 다른" 출력(이후 in-place 시 폭발)을 잡지 못하고, 변이된 **입력의 최종 상태**나 **list/dict 부작용**을 비교하지 않는 경우가 많다. 이번 퍼저는 이 셋을 모두 오라클에 넣었다.

### 1.3 이슈 트래커 현황 (2026년 9월 22일 기준)

- `gh api`로 2026-05-01 이후 `module: correctness (silent)` 이슈 365건을 수집(`fuzz/known_issues_silent_2026.tsv`). 월별 74/66/59/72/94건으로 증가세. 열림 171, 닫힘 194.
- 상위 신고자: ALinrunrun 31, xyt66665000 22, cuiliaomei-beep 16, bogatyy 15. 소수가 자동화 도구로 대량 신고하는 구조이며, 9월 21일 하루에 cuiliaomei-beep 7건 등 10건 이상.
- 별칭·뷰·in-place 키워드 이슈 41건. 대표: #197893(`x + 0`이 입력 자체를 반환, 열림), #197829(`x[1:] = x[:-1].clone()`의 clone 제거, 닫힘), #197489(storage offset 다른 clone을 no-op으로 제거, 닫힘), #198031(clone 제거로 커스텀 op가 같은 텐서를 두 번 받음), #197896(로컬 클래스 메서드 안의 list/dict 변이 누락), #196657(반복 변이된 별칭에서 functionalization 깨짐), #193760(`view(torch.bool).copy_` 데이터 손상), #191449(뷰에 `resize_`), #188084(expand 뷰의 index_put reinplace로 기울기 손상, 닫힘), #197887(입력 뷰가 backward에 저장될 때 in-place 검사 소실).
- 시사점: (1) 신고 속도 경쟁에서 이길 수 없으니 **체계적 오라클·근본 원인 분류**로 차별화, (2) 신고 전 이 TSV와 `gh` 검색으로 중복 확인 필수.

### 1.4 컴포넌트별 취약 지점 (문헌 + 이번 실증)

| 계층 | 취약 메커니즘 | 문헌 근거 | 이번 실증 |
|---|---|---|---|
| Dynamo | 자기 자신을 반환하는 무연산(`contiguous()`, `to()`)을 입력 별칭으로 인식 못함 → 메타 변이 가드 실패 | 그래프 캐싱 버그 4.3%, #129673, #197896 | F: 가드 생성 IndexError, "Guard failed on the same frame" |
| functionalization | 뷰 역변환(inverse) 공식 오류, 겹침 뷰 처리 | in-place 21.6%, #98143, #165409 | **A: unfold 역변환이 `unfold_backward`** |
| AOTAutograd | 별칭 입력(synthetic base)과 출력 뷰 재생성, 동적 형상에서 stride 추적 | #194747, #188133 | **B: ViewMeta를 잘못된 base에 재적용**, **G: 전치된 stride로 재생성** |
| Inductor | reinplace, no-op 제거, 버퍼 재사용, 변이 순서 | 레이아웃 충돌 12.1%, #197893, #197829, #195451, #183986 | **H1: 무연산 scatter·copy_ 별칭**, **H2: 입력 버퍼 reinplace**, **H3: clone 제거**, **I: dtype 뷰 왕복 순서 위반**, D: 변이가 죽은 버퍼에 스케줄(나이틀리에서 수정) |

---

## 2. 실증: 특화 퍼저 aliasfuzz

### 2.1 설계

- **프로그램**: 입력 2~3개(numel 24, 정수값 int64 또는 정수값 float64) + 30% 확률로 **별칭 입력** `ta = t0[1:]`, `t0.view(-1)`, `t0.transpose(0,1)`, `t0` 자체 + Python int `k` + `lst`, `dct`. 3~10문장을 가중 랜덤 생성. 문장 종류: 뷰 9종(slice, transpose/permute, view/reshape/flatten, squeeze, select/narrow, diagonal, unfold, expand, dtype view), 무연산 별칭(contiguous/detach/alias/view_as/`[...]`/`[:]`), 함수형 연산, in-place 원소 연산, setitem(슬라이스/인덱스/마스크), index_fill_/index_add_/index_copy_/scatter_add_, in-place 메타 변경(transpose_/unsqueeze_/squeeze_/t_), slice/select/diagonal_scatter, Python 부작용.
- **정의되지 않은 동작 회피**: in-place 대상과 저장소 뿌리가 같은 피연산자를 읽지 않음(부분 겹침 읽기-쓰기는 PyTorch가 `TOO_HARD`로 판정해 조용히 순서 의존 결과를 내므로 오탐 원인). 첫 스모크에서 실제로 이런 오탐 1건이 나와 규칙을 추가했다. 겹치는 unfold/expand는 읽기 전용, 자기 겹침 대입은 `.clone()`.
- **오라클 5종**: (1) 반환값 정확 비교(정수값이라 허용오차 0), (2) 함수가 변이한 **입력의 최종 상태**, (3) list/dict 부작용, (4) **별칭 프로브**: 반환 텐서에 서로 다른 상수를 in-place로 더한 뒤 다른 반환값·입력으로 전파되는 양상을 eager와 비교, (5) 데이터와 `k`를 바꿔 2회 호출(가드·캐시).
- **최소화**: 문장 제거 → 반환값 제거 → 별칭 입력 제거 → dynamic 해제의 탐욕적 델타 디버깅. 저장된 재현 파일에서 직접 최소화 가능.
- **비용**: 프로그램당 aot_eager 약 0.1초, Inductor CUDA 약 0.5초, Dynamo 전용 약 0.05초.

### 2.2 실행 환경

- Windows 11, i7-14700K, RTX 4070 Ti 12GB, Python 3.12
- torch 2.14.0+cu130 (안정판), triton-windows 3.8.0 → Inductor는 CUDA에서만(MSVC 부재로 Inductor CPU 불가)
- torch 2.15.0.dev20260921+cpu (나이틀리, AOTAutograd 계열 재확인용) 및 torch 2.15.0.dev20260921+cu130 + triton-windows 3.8.0 (나이틀리, Inductor 계열 재확인용, venv `pt2nightly_cu`)

### 2.3 실행 통계

| 실행 | 백엔드/디바이스 | 시간 | 프로그램 수 | 값 불일치 | 별칭 불일치 | compile_error | 무효(eager 오류) |
|---|---|---|---|---|---|---|---|
| 스모크 | eager(Dynamo만)/CPU | 15초 | 308 | 0 | 0 | 1 | 36 |
| 스모크 | aot_eager/CPU | 20초 | 216 | 5 | 0 | 3 | 28 |
| 스모크 | inductor/CUDA | 75초 | 172 | 1 | 7 | 1 | 23 |
| A | aot_eager/CPU | 15분 | 8,779 | 76 | 2 | 151 | 25 |
| B | inductor/CUDA | 21분(60건 도달 시 중단) | 2,657 | 45 | 15 | 52 | 5 |
| D | eager(Dynamo만)/CPU | 8분 | 8,951 | 0 | 0 | 9 | 14 |

- UB 회피 규칙 도입 후(A, B, D) 무효율은 0.2~0.3%, 스모크 때(규칙 전) 12~13%였다.
- 사례 자동 분류(`triage.py`): A 런 78건 = unfold 계열 62 + 별칭입력 계열 15 + G 1. B 런 60건 = unfold 22 + 별칭입력 11 + 알려진 #197893 11 + D 4 + H2 8 + H3 2 + 기타 2(int64↔int32 dtype 뷰 산술 1, `copy_` 결과가 입력 별칭 1, 최소화 진행 중).
- compile_error 샘플(각 런 25건 저장): 대부분 B의 크래시 쌍둥이(`incorrect out shape after application of ViewMeta sequence`, `shape '[18]' is invalid for input of size 24`, `Cannot view a tensor with shape ...`)이고, B 런에는 알려진 제한 `aot_autograd() does not yet handle input mutations on views with different dtypes`(#194747) 3건, D 런은 F(Dynamo 가드 IndexError 6, "Guard failed on the same frame" 3).
- 처리량: aot_eager 초당 약 10개, Inductor CUDA 초당 약 2개, Dynamo 전용 초당 약 19개.

### 2.4 발견 목록

#### A. [functionalization] 간격 있는 `unfold` 뷰를 통한 in-place 변이가 창 밖 base 원소를 0으로 덮어씀 — 신규, 조용한 오답, 고심각도

```python
def fn(x):
    v = x.unfold(1, 1, 5)   # 창 크기 1, 간격 5: 0열과 5열만
    v.add_(v)
    return x
# eager   : [[2, 2, 3, 4, 5, 12], ...]
# compiled: [[2, 0, 0, 0, 0, 12], ...]   <- 1~4열이 0
```

- 재현: aot_eager(CPU, CUDA), Inductor(CUDA), 정적·동적 형상, 2.14.0과 나이틀리 모두. 반환 없이 입력만 변이해도 호출자의 텐서가 손상.
- 근본 원인: `torch.func.functionalize` 단독 재현. functionalized 그래프가 `unfold_copy → add → unfold_backward(add, [4,6], 1,1,5) → copy_(x, unfold_backward)`. `unfold_copy_inverse`(aten/src/ATen/FunctionalInverses.cpp)가 미분 공식 `unfold_backward`를 역변환으로 쓰는데, 이 함수는 어떤 창에도 덮이지 않은 위치에 0을 놓는다. 겹치는 창(step < size)은 #165409로 이미 오류 처리되고 옛 합산 버그는 #98143(2023)으로 닫혔지만, **간격 있는 창(step > size)** 은 검사 없이 통과해 조용히 0을 쓴다.
- 수정 방향: 비겹침 경우 `as_strided_scatter(base, mutated_view, ...)`처럼 base 값을 보존하는 역변환 사용.
- 변형: `x.view(24).unfold(0, 8, 24)`에 `index_fill_`(8~23번 원소 0), `unfold(0,1,2)`의 별칭에 setitem(1행·3행 0).
- 파일: `fuzz/repro/repro_unfold_zero.py`, `evidence_A_functionalize.py`, 이슈 초안 `ISSUE_DRAFT_A_unfold.md`

#### B. [aot_autograd] 별칭 입력(synthetic base)의 뷰를 반환하면 잘못된 텐서에서 재생성 — 신규, 조용한 오답 + 크래시 쌍둥이

```python
def fn(t0, ta):          # ta = t0[1:], storage offset 6
    t0.add_(100)
    return ta[0]         # eager: t0[1] (offset 6) / compiled: t0[0] (offset 0)
def fn2(t0, ta):
    t0.add_(100)
    return ta.view(-1)   # AssertionError: incorrect out shape after application of ViewMeta sequence: (24,) (actual) vs (18,) (expected)
```

- 재현: aot_eager(CPU, CUDA), Inductor(CUDA), 정적·동적, 2.14.0과 나이틀리. 변이된 입력 자체는 정확하고 반환 뷰만 틀림. `ta` 자체 반환, `ta`를 변이, `t0[1]` 반환은 정상.
- 근본 원인: `torch/_functorch/_aot_autograd/functional_utils.py::gen_alias_from_base`가 출력의 ViewMeta 시퀀스(`ta` 기준으로 기록됨)를 synthetic base `t0`에 재적용. 형상이 호환되면 조용히 틀린 위치(offset 0), 호환되지 않으면 단언 오류. aot_eager 런의 compile_error 상당수(`Cannot view a tensor with shape ... as (24,)`, `shape '[18]' is invalid for input of size 24`, `maximum size for tensor at dimension 0 is 2 but size is 10`)가 같은 메커니즘.
- 파일: `repro_aliased_input.py`, `repro_aliased_input_hyp.py`, `evidence_B_variants.py`, 이슈 초안 `ISSUE_DRAFT_B_synthetic_base_offset.md`

#### D. [inductor] `x.add_(5); x.index_fill_(1, idx, -1); x[x > 0] = 2`에서 `index_fill_`이 사라짐 — 안정판 2.14.0 전용(나이틀리에서 수정됨), 조용한 오답

```python
# eager   : [[2, 2, -1, 2], ...]
# compiled: [[2, 2,  2, 2], ...]
```

- 재현: Inductor CUDA(정적·동적), 2.14.0. aot_eager는 CPU·CUDA 모두 정상 → 그래프는 옳고 스케줄링이 문제. 반환 없이도 재현, `x += 5`/`x[:, :] += 5`도 재현. 첫 변이 제거, 로컬 텐서(`x0.clone()`), 슬라이스 대입 `x[:, 2] = -1`, `masked_fill_`, 함수형 `where`로 바꾸면 사라짐.
- 생성 코드(`TORCH_LOGS=output_code`): 융합 커널 0이 `load(x); +5; >0; where`를 계산해 `buf2`와 입력 `arg0_1`에 저장(변이 전 값으로 마스크·대체값 계산). 커널 1(`index_fill_`)이 **그 뒤에** `buf2`의 2열에 -1을 쓰고, `del buf2`. 즉 변이될 버퍼의 pointwise 정의가 소비자에 인라인되고, 변이는 순서 간선을 잃고 죽은 버퍼에 쓰인다.
- 관련: #183986(expand된 self에 대한 index 연산, 닫힘)과 이웃하지만 다른 조건. 퍼저의 case_006(`neg_ → index_fill_ → 마스크 setitem`)도 같은 계열.
- **나이틀리 2.15.0.dev20260921+cu130에서는 모든 변형이 정상**이므로 이미 수정된 버그다. 새 이슈보다는 수정 커밋을 찾아 2.14.x 백포트 여부를 묻는 용도. 이 확인이 "컴포넌트 귀속 + 버전 교차 확인" 절차의 가치를 보여준다.
- 파일: `repro_indexfill_mask.py`, `repro_indexfill_mask2.py`, `inductor_code_D.txt`, 이슈 초안 `ISSUE_DRAFT_D_inductor_indexfill_mask.md`

#### G. [aot_autograd][dynamic shapes] `transpose` 뷰로 입력을 변이한 뒤 반환한 `t0.view(t0.shape)` 별칭이 전치된 stride로 재생성 — 신규, 조용한 오답, dynamic=True 전용

```python
def fn(t0):
    v2 = t0.transpose(1, 2)
    v3 = t0.view(t0.shape)     # eager stride (12, 4, 1)
    v2.masked_fill_(v2 > 2, 5)
    return v3                  # compiled stride (12, 1, 3): 원소 순서가 뒤바뀜, 2-D 에서는 값도 틀림
```

- 재현: aot_eager CPU, `dynamic=True`에서만, 2.14.0과 나이틀리. 변이된 입력은 정확. 변이 없음·직접 변이·`t0` 자체 반환은 정상.
- 해석: `(12, 1, 3)`은 `(2,4,3)` 연속 텐서를 다시 전치한 stride, 즉 functionalization이 만든 갱신 base(`transpose_copy(masked_fill(transpose_copy(t0)))`)의 레이아웃. 출력 별칭을 eager 뷰의 stride 대신 그 중간 텐서의 stride로 재생성한다.
- 파일: `repro_reshape_alias_dynamic.py`, 이슈 초안 `ISSUE_DRAFT_G_dynamic_alias_strides.md`

#### H1. [inductor] 무연산 `slice_scatter/select_scatter/diagonal_scatter(x, x.view)`가 `x` 자체를 반환 — 신규(#197893의 scatter 변형), 조용한 오답

```python
y = torch.diagonal_scatter(x, x.diagonal()); x.fill_(2); return y   # eager: 원래 값 / inductor: 전부 2
```

`select_scatter(x, x.select(0,1), 0, 1)`, `slice_scatter(x, x[1:3], 0, 1, 3)`도 동일. aot_eager 정상, 나이틀리 CUDA 재현. `remove_noop_ops`가 값 기준 무연산을 입력으로 치환해 별칭이 생기고, 이후 입력 변이가 함수형 결과를 덮어쓴다. 변형: `y = x.neg(); ...; y.copy_(x); return y`에서 반환된 `y`가 입력 `x`의 별칭이 됨(값은 맞고 별칭 프로브만 검출, 퍼저 case_035). 파일: `repro_inductor_alias_families.py`, `ISSUE_DRAFT_H1_inductor_noop_scatter_alias.md`

#### H2. [inductor] `y = slice_scatter(x, src, ...); x.zero_(); return y`가 0을 반환 — 신규(#195451과 같은 경로, 값까지 틀림), 조용한 오답

```python
y = torch.slice_scatter(x, src[1:3], dim=0, start=1, end=3); x.zero_(); return y   # inductor: 전부 0
```

`select_scatter` + `fill_`도 동일. aot_eager 정상, 나이틀리 CUDA 재현. scatter 결과를 입력 `x`의 버퍼에 reinplace한 뒤, 뒤따르는 입력 변이 write-back이 그 버퍼를 덮어쓴다. `y = x.clone(); y[1:3] = src[1:3]`으로 바꾸면 정상. 파일: `repro_inductor_alias_families.py`, `ISSUE_DRAFT_H2_inductor_scatter_reinplace_input.md`

#### H3. [inductor] `x.index_fill_(...); y = x.clone(); x.clamp_(-4, 4); return y`에서 `clone()`이 제거됨 — 신규, 조용한 오답

`index_fill_`을 빼거나 `x.add_(1)`로 바꾸면 정상이므로 `index_put` 로워링이 만든 버퍼의 추적 오류다. D는 나이틀리에서 고쳐졌지만 이 증상은 나이틀리 CUDA에서도 재현되므로 원인이 다르고 별도 이슈로 분리했다. 파일: `repro_inductor_alias_families.py`, `ISSUE_DRAFT_H3_inductor_indexfill_clone_dropped.md`

#### I. [inductor] dtype 뷰 왕복 + 중간 텐서 in-place + 이후 입력 변이 → 변이 후 값으로 재계산 — 신규, 조용한 오답

```python
y = x.view(torch.int32) * 2   # 입력의 dtype 뷰를 읽어 만든 새 텐서
y.sub_(-4)                    # 중간 텐서 in-place
x[:, 2:5] = 2                 # 그 뒤 입력 변이
return y.view(torch.int64)    # inductor: 변이 후 x 로 계산한 값 (4<<32)+8 이 나옴
```

네 요소가 모두 필요하다: 마지막 `view(torch.int64)`를 빼거나, `y`의 in-place를 함수형으로 바꾸거나, dtype 뷰를 없애면 정상. `fill_`이나 `unsqueeze` 뷰를 통한 변이도 동일. aot_eager 정상, 안정판·나이틀리 CUDA 재현. 최종 뷰 커널이 변이된 중간 버퍼를 읽지 않고 pointwise 정의를 인라인해 입력을 다시 읽으면서 입력 변이와의 순서 간선을 잃는 것으로 보인다(D와 같은 종류의 실수지만 D 수정 후에도 남음). 파일: `repro_dtypeview_rw_sweep.py`, `repro_dtypeview_read_after_write.py`, `ISSUE_DRAFT_I_inductor_dtypeview_inplace_order.md`

#### C. [inductor] int64 중간 텐서와 그 dtype 뷰(`view(torch.int32)`)를 함께 반환하면 별칭이 보존되지 않음 — 저심각도, 별칭 프로브만 검출

```python
v1 = t0.clone(); v4 = v1.view(torch.int32); v5 = v4.permute(0, 1); v5.clamp_(-1, 1)
return (v1, v5)   # eager: 같은 저장소 / compiled: 별도 버퍼 (값은 같음)
```

값 비교로는 잡히지 않고 별칭 프로브만 잡는다. 반환 후 사용자가 한쪽을 in-place로 바꾸면 eager와 달라진다. 안정판·나이틀리 CUDA 모두 재현. 관련 열린 이슈 #195451(출력 별칭 변경), #193760(dtype 뷰 손상). 신고 가치는 있으나 "의미 차이" 수준.

#### F. [dynamo] 자기 자신을 반환하는 무연산 뒤의 차원 수 변경 in-place가 가드 생성을 깨뜨림 — 크래시(조용하지 않음), #129673의 잔존

```python
v = x.contiguous()   # x 가 연속이면 v is x
v.unsqueeze_(2)      # IndexError: list index out of range  (symbolic_shapes.produce_guards_verbose, constraint_size[i])
```

`x.to(x.dtype)`, `x.to(x.device)` 뒤에도 동일. 직접 `x.unsqueeze_(0)`, `x.squeeze_(0)`, `x.t_()`는 정상이고 `x.requires_grad_()` 경유도 정상. 즉 Dynamo가 자기 자신을 반환하는 무연산을 입력의 별칭으로 인식하지 못해 입력 메타 변이 처리를 건너뛴다. 2024년 이슈 #129673(직접 `unsqueeze_` 크래시)은 2025년 4월 닫혔지만 이 경로는 남았다. 안정판·나이틀리 동일. Dynamo 전용 런 8,951개 중 9건(IndexError 6, `Guard failed on the same frame it was created` 3). 파일: `repro_dynamo_noop_alias_meta.py`, `repro_dynamo_meta_inplace.py`, `ISSUE_DRAFT_F_dynamo_rank_changing_inplace.md`

#### K. 알려진 버그 재발견: #197893 (`x + 0`, `x * 1`이 입력 자체를 반환)

Inductor 스모크 75초에서 별칭 프로브가 6번 검출. 9월 21일에 이미 신고된 열린 이슈라 생성기에서 해당 상수를 제외했다. 별칭 프로브 오라클이 이 유형을 값 비교 없이 잡는다는 검증 사례.

### 2.5 오탐과 그 처리

- 대상과 부분적으로 겹치는 메모리를 읽는 in-place(`v1.sub_(t2.narrow(2,0,1))`, `v1`은 `t2`의 별칭): eager는 겹침 판정이 `TOO_HARD`면 검사 없이 순서 의존 결과를 내고 컴파일은 수학적 기대값을 낸다. PyTorch 계약상 UB이므로 생성기에서 배제.
- `x + 0`/`x * 1` 상수 경로: 알려진 버그 재발견만 반복하므로 배제.
- 그 외 오탐은 관찰되지 않음(정수값 데이터로 허용오차 문제를 원천 제거한 효과).

---

## 2.6 수정 진행 상황 (2026-09-22, 저장소 https://github.com/wwoosshh/AI-accelerator-compiler)

버그별 추적 이슈를 저장소에 등록했고(#1 A, #2 B, #3 G, #4 F, #5 H1, #6 H2, #7 H3, #8 I, #9 D, #10 C), PyTorch main(dc0133e)을 클론해 수정을 시작했다. 패치와 검증 방법은 `fuzz/patches/README.md`.

| 대상 | 수정 위치 | 내용 | 검증 |
|---|---|---|---|
| H1·H2·H3 | `torch/_inductor/fx_passes/reinplace.py` `can_inplace` | 입력에 되쓰는 `copy_`가 이 연산의 결과를 되쓰는 것이 아니고 결과가 그 뒤에도 관측되면 입력 버퍼에 reinplace 금지 | 재현 전부 통과, 새 회귀 테스트(패치 전 실패·후 통과), 기존 테스트 회귀 없음, 정상 패턴 추가 할당 없음 |
| B | `torch/_functorch/_aot_autograd/input_output_analysis.py` `create_synthetic_base_metadata` | 병합된 입력의 출력 별칭은 원래 입력 기준 ViewMeta 를 버리고 출력 메타데이터로 `as_strided` 재생성 | 변형 7종·크래시 변형 통과, 새 회귀 테스트(패치 전 실패·후 통과) |
| F | `torch/_dynamo/variables/tensor.py` `call_method` | 자기 자신을 반환하는 무연산 결과는 같은 VariableTracker 로 유지 | 재현 9종 통과, 새 회귀 테스트(패치 전 IndexError·후 통과), Dynamo 표적 테스트 81 통과 |
| I | `torch/_inductor/graph.py` `mark_buffer_mutated` | 별칭 버퍼의 소비자도 변이 전에 실체화 | 변형 8종·퍼저 사례 통과, 새 회귀 테스트 통과, Inductor 표적 테스트 51 통과 |
| A | `aten/src/ATen/FunctionalInverses.cpp` `unfold_inverse` | `size <= step`이면 `as_strided_scatter`로 되쓰기 | 공식 Python 613 케이스 검증. C++ 검증은 Docker(Linux) 에서 CPU 전용 PyTorch 빌드로 진행 중(결과는 `fuzz/patches/README.md` 참조) |
| G | 원인 확정, 미수정 | 심볼릭 ViewMeta 폐기 + 폴백이 갱신 base 의 전치 stride 사용. 설계 결정 필요 | |
| C, C2, J | 미수정 | C2(동적 형상 별칭 전용), J(동적 형상 reshape 별칭 in-place 2회차 미반영)는 회귀 퍼징에서 새로 발견된 기존 버그 | |
| D | 불필요 | 나이틀리에서 이미 수정 | |

**수정 효과의 실측 검증**: 패치를 적용한 나이틀리 빌드에 퍼저를 다시 돌렸다. aot_eager 6,342개 프로그램에서 B 계열 0건(수정 전 78건 중 15건), Inductor 1,525개에서 H·D 계열 0건(수정 전 60건 중 14건). 남은 불일치는 A(C++ 미적용)와 새 발견 J·C2 뿐이었다.

---

## 3. 연구로 이어가기 위한 제언

1. **오라클이 논문의 핵심**: 별칭 프로브(반환 텐서 간·입력 간 별칭 구조 비교), 입력 변이 write-back 비교, 부작용 비교, 정수값 정확 비교의 조합은 기존 퍼저에 없다. 이를 형식화(별칭 그래프의 동형성 검사)하면 기여가 분명해진다.
2. **컴포넌트 귀속 자동화**: 같은 프로그램을 `eager → aot_eager → inductor` 순으로 돌려 어느 계층에서 처음 갈라지는지 자동 판정(이번 D가 그 예). functionalization 단독 재현(`torch.func.functionalize`)도 자동화 가능.
3. **생성기 확장**: autograd(뷰가 backward에 저장되는 #197887류), `resize_`/`set_`, 커스텀 op의 `mutates_args`, `torch.cond`/`while_loop` 안의 변이, DTensor. 각각 이슈 트래커에 열린 사례가 있어 수요가 확실하다.
4. **중복 관리**: 신고 전 `known_issues_silent_2026.tsv`와 `gh` 검색으로 확인. A·B·D는 이번 검색에서 중복이 없었다.
5. **신고는 사용자가 직접**: `fuzz/repro/ISSUE_DRAFT_*.md`(A, B, G, F, H1, H2, H3, I 8건은 나이틀리 재현 확인, D는 수정 커밋 확인 후 백포트 문의용)가 제출용 초안이다. 제출 전 `python -m torch.utils.collect_env` 출력을 붙이고, H1은 #197893에 댓글로, H2는 #195451에 댓글로 덧붙이는 편이 빠를 수 있다.

---

## 4. 참고 자료

- [Demystifying the Silence of Correctness Bugs in PyTorch Compiler (2026)](https://arxiv.org/abs/2604.08720)
- [TorchProbe](https://arxiv.org/abs/2310.20078), [WhiteFox](https://arxiv.org/abs/2310.15991), [OATest](https://arxiv.org/pdf/2511.18918), [Volta](https://arxiv.org/pdf/2511.12638), [Your Compiler is Backdooring Your Model](https://arxiv.org/pdf/2509.11173)
- PyTorch 이슈: [#197893](https://github.com/pytorch/pytorch/issues/197893), [#197829](https://github.com/pytorch/pytorch/issues/197829), [#197489](https://github.com/pytorch/pytorch/issues/197489), [#198031](https://github.com/pytorch/pytorch/issues/198031), [#165409](https://github.com/pytorch/pytorch/issues/165409), [#98143](https://github.com/pytorch/pytorch/issues/98143), [#183986](https://github.com/pytorch/pytorch/issues/183986), [#194747](https://github.com/pytorch/pytorch/issues/194747), [#188133](https://github.com/pytorch/pytorch/issues/188133), [#195451](https://github.com/pytorch/pytorch/issues/195451), [#193760](https://github.com/pytorch/pytorch/issues/193760)
- [State of torch.compile (ezyang, 2025-08)](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/)
