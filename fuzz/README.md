# aliasfuzz: torch.compile 별칭·뷰·in-place 특화 차등 퍼저

`torch.compile` 의 "조용한 오답" 버그 중 메모리 계열(in-place, 뷰/별칭, 입력 변이 write-back, Python 부작용)을
겨냥한 소형 프로그램 생성기 + 차등 오라클입니다. 2026-09-22 기준 torch 2.14.0 에서 약 21,000개 프로그램(45분)으로
나이틀리에서도 재현되는 신규 조용한 오답 7계열(A, B, G, H1, H2, H3, I), 안정판 전용 1건(D), 별칭 의미 차이 1건(C),
Dynamo 크래시 1군(F), 알려진 #197893 재발견을 얻었습니다. 자세한 결과는
`../연구1_torch.compile_정확성버그_심층조사_2026-09.md` 를, 이슈 초안은 `repro/ISSUE_DRAFT_*.md` 를 보세요.

## 환경

- venv: `C:\Users\s0105\venvs\pt2bug` (torch 2.14.0+cu130, triton-windows 3.8.0, Python 3.12)
- nightly 확인용 venv: `C:\Users\s0105\venvs\pt2nightly` (torch 2.15.0.dev20260921+cpu), `C:\Users\s0105\venvs\pt2nightly_cu` (torch 2.15.0.dev20260921+cu130 + triton-windows 3.8.0)
- Inductor CPU 백엔드는 MSVC(cl.exe)가 없어 사용 불가. Inductor 는 CUDA(RTX 4070 Ti)로, AOTAutograd 계층은
  `aot_eager` 백엔드(CPU)로 검사합니다.

## 실행

```bash
# AOTAutograd(functionalization, 뷰 재생성, 입력 변이) 계층: 빠름 (초당 ~10 프로그램)
C:/Users/s0105/venvs/pt2bug/Scripts/python.exe aliasfuzz.py --backend aot_eager --device cpu --minutes 15 --seed 11 --out results/A_aot_cpu

# Inductor(스케줄러, reinplace, 코드 생성) 계층: 초당 ~2 프로그램
C:/Users/s0105/venvs/pt2bug/Scripts/python.exe aliasfuzz.py --backend inductor --device cuda --minutes 25 --seed 21 --out results/B_ind_cuda

# Dynamo 만 (가드, 부작용 재생): 초당 ~20 프로그램
C:/Users/s0105/venvs/pt2bug/Scripts/python.exe aliasfuzz.py --backend eager --device cpu --minutes 8 --seed 31 --out results/D_dynamo_cpu

# 저장된 사례를 다시 최소화 (생성기가 바뀌어도 동작)
C:/Users/s0105/venvs/pt2bug/Scripts/python.exe aliasfuzz.py --minimize-file results/A_aot_cpu/case_001.py --min-budget 80

# 결과 자동 분류 (A unfold / B 별칭 입력 / C dtype 뷰 / 알려진 #197893 / 기타)
C:/Users/s0105/venvs/pt2bug/Scripts/python.exe triage.py results/A_aot_cpu results/B_ind_cuda
```

Windows 에서는 `TORCHINDUCTOR_COMPILE_THREADS=1` 을 권장합니다.

## 설계

- **프로그램**: 입력 텐서 2~3개(모두 numel 24, 정수값 int64 또는 정수값 float64) + 선택적으로 `ta = t0[1:]` 같은
  **별칭 입력** + Python int `k` + `lst`, `dct`. 3~10개 문장을 가중 랜덤으로 생성:
  뷰(slice, transpose/permute, view/reshape/flatten, squeeze, select/narrow, diagonal, unfold, expand, dtype view),
  무연산 별칭(contiguous/detach/alias/view_as/`[...]`), 함수형 연산, in-place 원소 연산, setitem(슬라이스/인덱스/마스크),
  index_fill_/index_add_/index_copy_/scatter_add_, in-place 메타 변경(transpose_/unsqueeze_/squeeze_/t_),
  slice_scatter/select_scatter/diagonal_scatter, Python 부작용(list/dict).
- **정의되지 않은 동작 회피**: in-place 대상과 저장소 뿌리가 같은 피연산자는 쓰지 않음(부분 겹침 읽기-쓰기는 PyTorch 계약상 UB).
  자기 겹침 대입은 `.clone()` 을 붙임. 겹치는 unfold/expand 는 읽기 전용.
- **오라클**: 값이 정수라 허용오차 0. (1) 반환값, (2) 변이된 입력 상태, (3) list/dict 부작용, (4) 별칭 프로브(반환 텐서에
  in-place 덧셈 후 전파 양상 비교), (5) 데이터·`k` 를 바꿔 2회 호출(가드/캐시).
- **최소화**: 문장 제거 → 반환값 제거 → 별칭 입력 제거 → dynamic 해제의 탐욕적 델타 디버깅.
- **재현 파일**: `case_XXX.py` 는 독립 실행 가능(`python case_XXX.py`), `case_XXX.json` 에 메타데이터.

## 파일

- `aliasfuzz.py` 퍼저 본체, `triage.py` 사례 분류
- `repro/` 이슈 제출용 독립 재현 스크립트와 영문 이슈 초안(`ISSUE_DRAFT_*.md`)
- `known_issues_silent_2026.tsv` 2026-05 이후 `module: correctness (silent)` 이슈 365건(중복 신고 방지용)
- `results/` 실행 로그, 사례, 최소화 결과, `summary.json`
