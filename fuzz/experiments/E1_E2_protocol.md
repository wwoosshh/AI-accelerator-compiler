# E1/E2. 같은 예산에서 NNSmith 대 aliasfuzz

## 목적
"별칭·in-place 프로그램 공간 + 입력 상태·별칭 오라클" 이 기존 연산자 그래프 퍼저 대비 무엇을 더 잡는지 같은 조건에서 측정한다. E3(오라클 절제)가 오라클 기여를 보이고, E1/E2 는 생성기 기여를 보인다.

## 조건 (양쪽 동일)
- 소프트웨어: torch 2.14.0+cu130 (패치 없음, venv `pt2bug`), triton-windows 3.8.0, Python 3.12, Windows 11
- 하드웨어: i7-14700K, RTX 4070 Ti 12GB. 두 실험을 동시에 실행해 자원 경쟁 조건도 동일
- 대상: `torch.compile(backend="inductor")`, CUDA, 기본 모드
- 예산: 각 20분 wall-clock, `TORCHINDUCTOR_COMPILE_THREADS=1`

## 도구 설정
- **E1 NNSmith** (GitHub main, 2026-09-22 설치; PyPI 0.1.0 에는 `pt2` 백엔드가 없음): `nnsmith.cli.fuzz fuzz.time=20m model.type=torch backend.type=pt2 backend.target=cuda mgen.max_nodes=8`. 생성기는 연산자 그래프(순수 함수형 op 8개 이하), 오라클은 **같은 디바이스(CUDA) eager** 참조값과 pt2 출력(`fx.symbolic_trace` 후 `torch.compile(fullgraph=True, backend=inductor)`)의 `allclose(rtol=1e-2, atol=1e-3)`. (처음에는 CPU eager 와 비교한다고 적었으나 `backend.target=cuda` 면 참조값도 CUDA 에서 계산된다 — 결과 문서에서 정정.)
- **E2 aliasfuzz**: `aliasfuzz.py --backend inductor --device cuda --minutes 20 --seed 71 --max-cases 1000`. 생성기는 별칭·뷰·in-place·부작용 프로그램(3~10문장), 오라클은 같은 디바이스 eager 대비 정수값 정확 비교 + 입력 상태 + 부작용 + 별칭 프로브 + 2회 호출. 알려진 #197893 재발견 패턴(`x + 0`, `x * 1`)은 생성기에서 제외되어 있음.

## 측정 항목
1. 생성·실행한 프로그램 수, 무효/생성 실패 수
2. 도구가 보고한 불일치 수
3. **재검증 후 남는 불일치 수와 그 성격** (NNSmith 보고는 NNSmith 자신의 컴파일 경로로 다시 컴파일해 재현 여부를 확인하고, 불일치 출력을 dtype·스케일·불연속 연산 기준으로 분류 + 1 ulp 입력 섭동 검사로 수치적 불안정성을 가른다)
4. 고유 근본 원인(계열) 수와 종류. aliasfuzz 는 `triage.py` 로 분류, NNSmith 는 연산 목록과 오차 크기로 수작업 분류
5. 컴파일 오류(크래시) 수

## 공정성 메모
- NNSmith 의 `max_nodes=8` 은 aliasfuzz 의 3~10문장과 비슷한 크기. NNSmith 기본 예제도 5~10 노드를 쓴다.
- NNSmith 는 float 연산 그래프라 tolerance 오라클이 불가피하고, aliasfuzz 는 정수값 데이터로 정확 비교가 가능하다. 이는 도구 설계 차이의 일부이므로 그대로 두고 3번 항목으로 교차 디바이스 효과만 분리한다.
- NeuRI 는 NNSmith 생성기 확장이라 프로그램 공간 성격이 같고, TorchProbe 는 코드 비공개, WhiteFox 는 LLM 생성 비용 때문에 제외했다.

## 실행 기록

- **1차 (2026-09-22 12:20~12:40)**: E1 정상 완료. E2 는 1228번째 프로그램에서 중단됐다. 별칭 입력이 `ta = t0`(같은 객체) 형태일 때 `ta.transpose_()` 가 `t0` 의 형상도 바꾸는데 생성기가 `t0` 의 형상을 갱신하지 않아 `t0.index_fill_` 에 범위 밖 인덱스가 들어갔고, CPU 에서는 IndexError 로 '무효' 처리되던 것이 CUDA 에서는 device-side assert 가 되어 컨텍스트가 오염됐다. 이후 333만 회의 반복이 모두 즉시 '무효' 로 집계되어 실효 예산은 약 8.7분(1,224개 프로그램)에 그쳤다. 이 부분 결과는 참고용으로만 보고한다.
- **하네스 수정**: (1) 생성기가 같은 객체를 가리키는 두 이름의 형상을 함께 갱신(`Gen.identity_twins`), (2) `invalid` 사유 집계와 CUDA 컨텍스트 오염 감지(`cuda_healthy`, 종료 코드 3), (3) 예산 안에서 새 프로세스로 재시작해 부분 결과를 합치는 드라이버 `run_budget.py`. 스모크 테스트(42초, 97개 프로그램, 무효 0) 후 재실행.
- **2차 (2026-09-22 13:00~13:20, 본 결과)**: E1 과 E2 를 같은 시각에 시작해 20분 동시 실행. E2 는 `run_budget.py --minutes 20 --seed 72 -- --backend inductor --device cuda --max-cases 1000`, E1 은 위와 같은 명령에 `fuzz.root=E1b_nnsmith_pt2_20m`.

## 결과
`E1_E2_results.md` 에 기록.
