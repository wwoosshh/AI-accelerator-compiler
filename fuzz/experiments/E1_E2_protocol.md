# E1/E2. 같은 예산에서 NNSmith 대 aliasfuzz

## 목적
"별칭·in-place 프로그램 공간 + 입력 상태·별칭 오라클" 이 기존 연산자 그래프 퍼저 대비 무엇을 더 잡는지 같은 조건에서 측정한다. E3(오라클 절제)가 오라클 기여를 보이고, E1/E2 는 생성기 기여를 보인다.

## 조건 (양쪽 동일)
- 소프트웨어: torch 2.14.0+cu130 (패치 없음, venv `pt2bug`), triton-windows 3.8.0, Python 3.12, Windows 11
- 하드웨어: i7-14700K, RTX 4070 Ti 12GB. 두 실험을 동시에 실행해 자원 경쟁 조건도 동일
- 대상: `torch.compile(backend="inductor")`, CUDA, 기본 모드
- 예산: 각 20분 wall-clock, `TORCHINDUCTOR_COMPILE_THREADS=1`

## 도구 설정
- **E1 NNSmith** (GitHub main, 2026-09-22 설치; PyPI 0.1.0 에는 `pt2` 백엔드가 없음): `nnsmith.cli.fuzz fuzz.time=20m model.type=torch backend.type=pt2 backend.target=cuda mgen.max_nodes=8`. 생성기는 연산자 그래프(순수 함수형 op 8개 이하), 오라클은 CPU eager 출력과 pt2(CUDA) 출력의 `allclose(rtol=1e-2, atol=1e-3)`.
- **E2 aliasfuzz**: `aliasfuzz.py --backend inductor --device cuda --minutes 20 --seed 71 --max-cases 1000`. 생성기는 별칭·뷰·in-place·부작용 프로그램(3~10문장), 오라클은 같은 디바이스 eager 대비 정수값 정확 비교 + 입력 상태 + 부작용 + 별칭 프로브 + 2회 호출. 알려진 #197893 재발견 패턴(`x + 0`, `x * 1`)은 생성기에서 제외되어 있음.

## 측정 항목
1. 생성·실행한 프로그램 수, 무효/생성 실패 수
2. 도구가 보고한 불일치 수
3. **같은 디바이스 재검증 후 남는 불일치 수** (NNSmith 는 CPU eager 와 비교하므로 GPU-CPU 부동소수점 차이가 섞임 → 저장된 모델을 CUDA eager 대 CUDA 컴파일로 다시 비교)
4. 고유 근본 원인(계열) 수와 종류. aliasfuzz 는 `triage.py` 로 분류, NNSmith 는 연산 목록과 오차 크기로 수작업 분류
5. 컴파일 오류(크래시) 수

## 공정성 메모
- NNSmith 의 `max_nodes=8` 은 aliasfuzz 의 3~10문장과 비슷한 크기. NNSmith 기본 예제도 5~10 노드를 쓴다.
- NNSmith 는 float 연산 그래프라 tolerance 오라클이 불가피하고, aliasfuzz 는 정수값 데이터로 정확 비교가 가능하다. 이는 도구 설계 차이의 일부이므로 그대로 두고 3번 항목으로 교차 디바이스 효과만 분리한다.
- NeuRI 는 NNSmith 생성기 확장이라 프로그램 공간 성격이 같고, TorchProbe 는 코드 비공개, WhiteFox 는 LLM 생성 비용 때문에 제외했다.

## 결과
(실행 후 `E1_E2_results.md` 에 기록)
