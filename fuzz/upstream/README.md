# 업스트림 제출 상태

**2026-09-22 제출 완료**: H → 이슈 pytorch/pytorch#198094, PR #198096. B → 이슈 #198095, PR #198097. F → 이슈 #198100, PR #198103. I → 이슈 #198101, PR #198104. A → 이슈 #198102, PR #198105. 기존 이슈 #197893, #195451 에 연결 댓글. J 는 수정 브랜치가 없어 이슈만 준비됨(미제출). 상세는 `SUBMITTED.txt`.


포크 https://github.com/wwoosshh/pytorch 에 아래 5개 브랜치가 최신 main(6b5c7ad, 2026-09-22) 위에 커밋 1개씩으로 올라가 있습니다. 로컬 클론은 `C:\Users\s0105\src\pytorch`.

| 순서 | 브랜치 | 내용 | PR 본문 | 이슈 본문 |
|---|---|---|---|---|
| 1 | `fix/inductor-reinplace-input-overwrite` | H1·H2·H3 | `PR_1_inductor_reinplace.md` | 기존 #197893, #195451 에 댓글(`COMMENTS_existing_issues.md`) + 새 이슈 권장 |
| 2 | `fix/aot-synthetic-base-output-alias` | B | `PR_2_aot_synthetic_base.md` | `ISSUE_B_synthetic_base.md` |
| 3 | `fix/dynamo-noop-method-returns-self` | F | `PR_3_dynamo_noop_self.md` | `ISSUE_F_dynamo_noop_alias.md` |
| 4 | `fix/inductor-realize-alias-consumers-before-mutation` | I | `PR_4_inductor_alias_consumers.md` | `ISSUE_I_inductor_dtype_view_order.md` |
| 5 | `fix/functionalization-unfold-inverse` | A (C++, CI 컴파일 필요) | `PR_5_functionalization_unfold.md` | `ISSUE_A_unfold.md` |
| 6 | `fix/inductor-noop-copy-of-mutated-input` (f01f746, main 00ec4ca 기반) | L: `dst[0:, :] = src[0:, :]; src.add_(1)` 이 갱신된 src 를 복사 (비교실험 E2 에서 발견, 패치 0006) | `PR_6_inductor_noop_copy_of_mutated_input.md` | `ISSUE_L_noop_copy_of_mutated_input.md` — **제출 전, 승인 필요** |
| - | (미수정) | J | - | `ISSUE_J_inductor_dynamic_index_add.md` |

`collect_env` 출력은 `../repro/collect_env_stable_2.14.0.txt`, `../repro/collect_env_nightly_2.15.0.dev20260921.txt`.

## 실행 순서

1. 이슈 먼저 (본문 파일의 Versions 아래에 collect_env 출력을 붙여도 좋음):

```bash
cd fuzz/upstream
gh issue create --repo pytorch/pytorch --title "[functionalization] In-place mutation through a gapped unfold view (step > size) zeroes every base element not covered by a window under torch.compile" --body-file ISSUE_A_unfold.md
gh issue create --repo pytorch/pytorch --title "[aot_autograd] Output that is a view of an aliased input (synthetic base) is regenerated from the wrong tensor: wrong values or ViewMeta shape assertion" --body-file ISSUE_B_synthetic_base.md
gh issue create --repo pytorch/pytorch --title "[dynamo] x.contiguous().unsqueeze_(d) / x.to(x.dtype).unsqueeze_(d) crashes guard creation with IndexError (regression of #129673 through no-op aliases)" --body-file ISSUE_F_dynamo_noop_alias.md
gh issue create --repo pytorch/pytorch --title "[inductor] y = x.view(torch.int32) * 2; y.sub_(-4); x[:, 2:5] = 2; return y.view(torch.int64) computes y from the mutated x" --body-file ISSUE_I_inductor_dtype_view_order.md
gh issue create --repo pytorch/pytorch --title "[inductor][dynamic shapes] index_add_ through a flat reshape alias of an input is dropped when the input was assigned to earlier (dynamic=True only)" --body-file ISSUE_J_inductor_dynamic_index_add.md
```

L (2026-09-22 추가; 최종 본문은 `ISSUE_L_final_body.md`(collect_env 첨부) / `PR_6_final_body.md`, PR 본문의 `Fixes #<L issue number>` 를 이슈 번호로 바꾼 뒤 PR):

```bash
cd fuzz/upstream
gh issue create --repo pytorch/pytorch --title "$(cat ISSUE_L_final_title.txt)" --body-file ISSUE_L_final_body.md
gh pr create --repo pytorch/pytorch --base main --head wwoosshh:fix/inductor-noop-copy-of-mutated-input --title "[inductor] remove_noop_ops: keep a copy of an input that is mutated before the copy's user runs" --body-file PR_6_final_body.md
gh pr comment <PR번호> --repo pytorch/pytorch --body '@pytorchbot label "release notes: inductor"'
```

2. PR 본문의 `Fixes #TBD` 를 위에서 받은 이슈 번호로 바꾼 뒤 PR:

```bash
gh pr create --repo pytorch/pytorch --base main --head wwoosshh:fix/inductor-reinplace-input-overwrite --title "[inductor] Do not reinplace into a graph input that a later, unrelated mutation overwrites" --body-file PR_1_inductor_reinplace.md
gh pr create --repo pytorch/pytorch --base main --head wwoosshh:fix/aot-synthetic-base-output-alias --title "[aot_autograd] Regenerate output aliases of merged (synthetic-base) inputs from metadata" --body-file PR_2_aot_synthetic_base.md
gh pr create --repo pytorch/pytorch --base main --head wwoosshh:fix/dynamo-noop-method-returns-self --title "[dynamo] Keep tracking self when a non-mutating method returns the tensor itself" --body-file PR_3_dynamo_noop_self.md
gh pr create --repo pytorch/pytorch --base main --head wwoosshh:fix/inductor-realize-alias-consumers-before-mutation --title "[inductor] Realize consumers of aliasing buffers before a mutation of the aliased buffer" --body-file PR_4_inductor_alias_consumers.md
gh pr create --repo pytorch/pytorch --base main --head wwoosshh:fix/functionalization-unfold-inverse --title "[functionalization] unfold inverse: keep base elements that no window covers" --body-file PR_5_functionalization_unfold.md
```

3. 첫 PR 뒤 CLA 봇 링크에서 서명. Inductor 테스트 워크플로는 리뷰어가 `ciflow/inductor` 라벨을 붙여야 돌아가므로 댓글로 요청.
   - **주의**: PyTorch 의 EasyCLA 는 커밋의 `Co-authored-by:` 트레일러에 적힌 사람도 CLA 서명자로 확인한다. GitHub 계정에 대응하지 않는 `Co-Authored-By: Claude ... <noreply@anthropic.com>` 같은 줄이 있으면 검사가 실패하므로 PyTorch 용 커밋에는 넣지 않는다(2026-09-22 에 5개 브랜치 모두 제거·강제 푸시하고 `/easycla` 재요청).
   - `release notes:` 라벨은 `@pytorchbot label "release notes: inductor"` 같은 댓글로 누구나 요청 가능(AOTAutograd/functionalization 은 `release notes: composability`).

4. 리뷰 반영은 같은 브랜치에 커밋 추가 후 `git push fork <branch>`. main 과 충돌 시 `git fetch origin main && git rebase origin/main && git push -f fork <branch>` (클론이 얕으므로 필요하면 `git fetch --unshallow origin` 먼저).
