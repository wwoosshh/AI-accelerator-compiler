# E1/E2 결과: 같은 20분 예산에서 NNSmith 대 aliasfuzz (2026-09-22)

프로토콜과 조건은 `E1_E2_protocol.md`. 아래 "본 결과" 는 두 도구를 같은 시각(13:00)에 시작해 20분 동시 실행한 2차 실행이다. 1차 실행은 aliasfuzz 쪽 하네스 결함으로 8.7분에 중단되어 참고용으로만 뒤에 적었다.

## 요약

1. **같은 20분에서 aliasfuzz 는 값이 달라지는 조용한 오답 30건(고유 근본 원인 6계열: A, B, D, H2, H3, L)과 별칭 전용 불일치 4건을 찾았고, NNSmith 는 불일치 34건을 보고했지만 전부 부동소수점 수치 차이(정수 동률·fp16 중간 반올림·불연속 연산 증폭)였고 의미 버그는 0건이었다.**
2. aliasfuzz 의 34건 중 1건은 **이번 실험에서 처음 발견된 계열 L**(입력 간 복사 뒤 원본을 in-place 변이하면 Inductor 가 복사를 변이 뒤로 보냄)이다. 원인을 `remove_noop_ops` 로 특정해 수정 패치 0006 과 회귀 테스트를 만들었다(`../patches/README.md`).
3. NNSmith 보고의 정체를 가리기 위해 NNSmith 자신의 컴파일 경로(`fx.symbolic_trace` + `torch.compile(fullgraph=True)`)로 재검증하고, 출력 dtype·스케일 기준 분류와 1 ulp 입력 섭동 검사를 했다. 1차 38건 중 30건, 2차 34건 중 29건은 입력의 마지막 비트만 흔들어도 NNSmith 허용오차를 넘는 불안정한 그래프였다.

## 본 결과 (2차, 13:00~13:20 동시 실행)

| 항목 | E1 NNSmith (pt2, CUDA, max_nodes=8) | E2 aliasfuzz (inductor, CUDA, seed 72) |
|---|---|---|
| 생성·실행한 프로그램 | 1,413 (생성 실패 76 별도) | 2,737 (무효 4: dtype 뷰 stride·view 호환 오류, 모두 생성기 쪽) |
| 도구가 보고한 불일치 | 34 (+ 실행 예외 2) | 34 (값 29, 별칭 전용 5) + 컴파일 오류 54 |
| 재검증 후 남는 불일치 | 34 (NNSmith 컴파일 경로로 전부 재현) | 34 (모두 결정적 재현) |
| 그중 **의미 버그**(정확히 같아야 할 값이 다름) | **0** — 정수 출력 동률 8, 저정밀(fp16, ≤32 ulp) 14, 불연속 연산 증폭 12 | **30** — A 16, B 8, D 2, H2 2, H3 1, **L 1** |
| 별칭 전용 불일치(값은 같음) | - | 4 — C(dtype 뷰) 1, C2(동적 형상 중간 텐서 뷰) 1, 확장(expand) 뷰 출력 실체화 1, cumsum 중간 텐서 뷰 별칭 소실 1 |
| 고유 근본 원인 수 | 0 | 값 6계열 + 별칭 전용 3형태 |
| 신규 발견 | 0 | L (Inductor `remove_noop_ops`; 패치 0006, 회귀 테스트, 포크 브랜치) |
| 크래시 | 2 — Dynamo 가 NNSmith 모델 코드의 generator 를 추적하지 못함(하네스 문제, gb0003) | 54 — 샘플 25건 중 B 계열 ViewMeta 단언 14, expand 크기 오류 2, "aot_autograd() does not yet handle input mutations on views with different dtypes" 2, 기타 7 |
| 검출 오라클 (E2) | (출력 tolerance 비교 하나) | 반환값 24, 입력 상태 비교로만 5, 별칭 프로브로만 5, 2회차 호출로만 0 |
| 처리량 | 71 프로그램/분 | 137 프로그램/분 (프로그램당 2회 호출 + 프로브 포함) |

E2 의 계열 분류는 `triage.py` + `e3_oracle_ablation.py` 규칙에 수작업 확인을 더한 것이다. 규칙이 '기타' 로 남긴 7건을 패치 적용 나이틀리(0001/0002/0004/0005/0006)에서 다시 돌려 귀속했다: 2건은 0001(H2: `select_scatter`/`diagonal_scatter` 결과가 나중에 덮어써지는 입력에 reinplace), 1건은 0001(H3: `clone` 뒤 원본 변이), 1건은 0006(L: `t1[:, :] = t2[:, :]; v1 = t2[-1, 1:3:2]; v1.zero_()` 에서 t1 이 0 을 받음), 3건은 별칭 전용으로 나이틀리에서도 남음(값은 같음). 전체 분석 출력은 `../results/E2b_analysis.txt`, 사례는 `../results/E2b_aliasfuzz_20m/part_0/`.

## 1차 실행 (참고, 12:20~12:40)

| 항목 | E1 NNSmith | E2 aliasfuzz (seed 71) |
|---|---|---|
| 프로그램 | 1,515 (생성 실패 76) | 1,224 — **1228번째에서 중단**(아래 참조), 이후 333만 회는 즉시 무효 |
| 보고 불일치 | 38 (+ 예외 2) | 22 (값 18, 별칭 4) + 컴파일 오류 24 |
| 의미 버그 | 0 — 정수 10, 저정밀 12, fp16 넘침 1, 불연속 증폭 13, 큰 차이 2(아래에서 둘 다 설명) | 22 — A 14, B 5, D 1, H2 1, L 1 |

중단 원인: 별칭 입력이 `ta = t0`(같은 객체)일 때 `ta.transpose_()` 가 `t0` 의 형상도 바꾸는데 생성기가 `t0` 쪽 형상을 갱신하지 않아 `t0.index_fill_` 에 범위 밖 인덱스가 들어갔다. CPU 라면 IndexError 로 '무효' 처리되고 끝나지만 CUDA 에서는 device-side assert 가 되어 컨텍스트가 오염되고 이후 모든 프로그램이 즉시 '무효' 로 집계됐다. 수정: 생성기가 같은 객체를 가리키는 두 이름의 형상을 함께 갱신(`Gen.identity_twins`), `invalid` 사유 집계, CUDA 오염 감지 시 종료 코드 3, 예산 안에서 새 프로세스로 재시작하는 드라이버 `run_budget.py`. 2차 실행에서는 재시작이 필요하지 않았다(무효 4건은 모두 일반 RuntimeError).

## NNSmith 불일치 68건의 정체

NNSmith 의 pt2 백엔드는 참조값을 **같은 디바이스(CUDA) eager** 로 계산하고, 모델을 `torch.fx.symbolic_trace` 한 뒤 `torch.compile(traced, fullgraph=True, backend="inductor")` 해서 `allclose(rtol=1e-2, atol=1e-3)` 로 비교한다(프로토콜 문서의 'CPU eager' 서술은 잘못이었고 정정했다). 재검증 스크립트 `reverify_nnsmith_same_device.py` 는 같은 경로로 다시 컴파일해 불일치 출력을 분류한다.

| 분류 | 기준 | 1차 (38) | 2차 (34) |
|---|---|---|---|
| int | 정수/불 출력만 다름 — argmin/argmax 동률, 반올림 경계 | 10 | 8 |
| lowprec | 최대 절대 오차가 출력 스케일의 32 ulp 이하(fp16 3.1%, fp32 3.8e-6) | 12 | 14 |
| overflow | fp16 에서 한쪽만 inf/nan | 1 | 0 |
| discontinuous | 그 이상 차이지만 그래프에 Ceil/Floor/Round/ArgMin/비교/Where/정수 캐스트가 있어 미세한 반올림이 증폭됨 | 13 | 12 |
| LARGE(수작업 확인) | 위 어디에도 해당 안 함 | 2 | 0 |
| trace | 추적된 모듈이 컴파일 없이도 다름(하네스) | 0 | 0 |

**1 ulp 입력 섭동 검사**(`nnsmith_condition_test.py`): float 입력의 각 원소를 1 ulp 만큼 무작위 부호로 흔든 뒤 eager 만 다시 실행해도 NNSmith 허용오차를 넘는 그래프가 1차 30/38, 2차 29/34 였다. 나머지 8+5건은 섭동이 닿지 않는 **상수 텐서**(NNSmith `Constant` 연산)가 지배적인 입력이거나 정수 경계 한 칸 차이(최대 차이 1)였다.

수작업으로 확인한 두 LARGE 사례(1차):

- **#21** (`leaky_relu(c) @ x → atan → tan`, 전부 fp16): matmul 값 691~906 의 `atan` 은 fp16 에서 전부 같은 값 1.5693359375 로 반올림되고, 그 `tan` 은 모두 684.5 가 된다(eager 출력이 상수). Inductor 는 `atan`·`tan` 연쇄를 fp32 로 계산해 한 번만 반올림하므로 `tan(atan(x)) ≈ x` 로 691~906 이 나온다. 컴파일 쪽이 수학적으로 더 정확하고, 차이는 fp16 중간 반올림에서 온다.
- **#10** (`Linear` 출력 1321~1606 에 `sin`, 이어서 Conv2d): 6만여 항을 더하는 fp32 내적의 누적 순서 차이(상대 1e-6 정도)가 절대 1e-3 의 인자 차이가 되고, 크기 1500 의 인자에 대한 `sin` 은 그 차이를 그대로 출력으로 옮긴다. 이후 1092 채널 합성곱이 더해 최대 0.107(스케일 141)이 됐다. 입력 섭동 검사가 잡지 못한 이유는 `Linear` 의 입력이 상수 텐서라서다.

즉 NNSmith 의 보고는 "eager 가 연산마다 fp16 으로 반올림하는 반면 Inductor 는 fp32 로 계산한다" 는 PyTorch 가 허용하는 정밀도 차이와, 그것이 불연속 연산을 지나며 커진 것들이다. 우리가 겨냥한 **의미 버그**(별칭·in-place·상태가 어긋나 정확히 같아야 할 정수값이 다른 것)는 한 건도 없었다.

## 공정성과 한계

- 프로그램 크기는 비슷하게 맞췄다(NNSmith 노드 8개 이하, aliasfuzz 3~10문장). 두 도구는 같은 GPU 에서 동시에 돌았으므로 자원 경쟁 조건도 같다.
- aliasfuzz 는 정수값·작은 실수 데이터로 정확 비교가 가능하고 NNSmith 는 fp16 연산 그래프라 tolerance 비교가 불가피하다. 이는 도구 설계의 일부지만, NNSmith 가 보고한 불일치를 우리가 전부 "수치 차이" 로 판정한 데에는 분류 규칙(32 ulp, 불연속 연산 목록)이 개입한다. 규칙과 사례별 근거는 `../results/E1_nnsmith/` 의 출력에 남겼다.
- aliasfuzz 생성기는 알려진 #197893 재발견 형태(`x + 0`, `x * 1`)를 제외하므로 K 계열이 빠져 있고, 시드 하나(72)의 결과다. 1차(seed 71, 8.7분)에서도 같은 계열 분포(A, B, D, H2, L)가 나왔다.
- NNSmith 는 순수 함수형 연산자 그래프 공간을 탐색하므로 별칭·in-place 계열이 나오지 않는 것은 설계상 당연하다. 이 실험이 보여주는 것은 "그 공간을 탐색하는 도구가 없다면 이 6계열은 발견되지 않는다" 는 점이며, E3 는 그 공간 안에서도 출력값 비교만으로는 30% 를 놓친다는 점을 보인다.

## 재현

```bash
# E2 (fuzz/ 에서)
C:/Users/s0105/venvs/pt2bug/Scripts/python.exe run_budget.py --minutes 20 --out results/E2b_aliasfuzz_20m --seed 72 -- --backend inductor --device cuda --max-cases 1000
C:/Users/s0105/venvs/pt2bug/Scripts/python.exe triage.py results/E2b_aliasfuzz_20m/part_0
C:/Users/s0105/venvs/pt2bug/Scripts/python.exe experiments/e3_oracle_ablation.py results/E2b_aliasfuzz_20m/part_0
C:/Users/s0105/venvs/pt2bug/Scripts/python.exe experiments/show_other_cases.py results/E2b_aliasfuzz_20m/part_0
# E1 (C:/Users/s0105/src/nnsmith_exp 에서)
C:/Users/s0105/venvs/pt2bug/Scripts/python.exe -m nnsmith.cli.fuzz fuzz.time=20m model.type=torch backend.type=pt2 backend.target=cuda fuzz.root=E1b_nnsmith_pt2_20m mgen.max_nodes=8 debug.viz=false
C:/Users/s0105/venvs/pt2bug/Scripts/python.exe <fuzz>/experiments/reverify_nnsmith_same_device.py E1b_nnsmith_pt2_20m
C:/Users/s0105/venvs/pt2bug/Scripts/python.exe <fuzz>/experiments/nnsmith_condition_test.py E1b_nnsmith_pt2_20m
```

파일: `../results/E2_aliasfuzz_20m/`(1차), `../results/E2b_aliasfuzz_20m/`(2차, `summary.json` 합산), `../results/E2b_analysis.txt`, `../results/E1_nnsmith/`(NNSmith 재검증·조건수 검사 출력; 원본 보고 폴더는 `C:\Users\s0105\src\nnsmith_exp`), 계열 L 재현 `../repro/repro_copy_then_view_indexfill.py`.
