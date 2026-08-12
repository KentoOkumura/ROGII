# exp397_prefix_gr_agreement_adaptive_sigma_exact_hmm

## 状態

- ルート: pf_beam
- 状態: stage_0_completed_guard_failed_closed
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-25
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`

## 仮説

known prefixでhorizontal GRとtypewell GRのshape一致度が低いwellでは、exp209のGaussian GR
emissionが誤ったmodeを強く固定しやすい。exp209のtrusted base scaleは維持し、一致度が低いwellだけ
scaleを `1.3` 倍してevidenceを弱めれば、良好wellを変えずにexact-HMMのtail errorを減らせる可能性がある。

## 変更点

- known prefixのraw finite GR pairでwell単位Pearson相関 `rho_gr` を計算する。
- `rho_gr >= 0.50` は係数 `1.0`、`rho_gr < 0.50` は係数 `1.3`。
- finite pair 64未満、低分散、nonfinite相関はtrusted parentへno-opの `1.0`。
- exp209の `[10, 60]` clip後の `sigma_gr` に係数を1回だけ掛け、再clipしない。
- `1.0` wellはsaved exp209 predictionを再利用し、`1.3` wellだけexact HMMを再計算する。
- その他のemission、state、transition、prior、outputはexp209から変えない。

## 検証方針

- Stage 0: 773 wellsのtruth-free agreement監査、HMM 0。coverage、係数非退化、full-prefixと
  last-512-row tailの安定性をAND gateで判定する。
- Stage 1: Stage 0全PASSと別承認後だけ1 variant / 5 reporting folds /
  最大773 exact-HMM well-runs、model・booster・PF・Beam・control再実行各0。
- primary: exp209比direct RMSE gain `>=0.05 ft`、4/5 folds。
- guard: changed group RMSE / p95 / 改善well率、worst well、raw observed/missing、高missing、
  1000+、hidden-like 2面、fixed LikPF 50:50をすべて確認する。
- leakage: agreement、係数、candidate predictionをunknown-suffix truth読込前にSHA freezeする。

## 実行入口

- train notebook scaffold:
  `exp397_prefix_gr_agreement_adaptive_sigma_exact_hmm_train.ipynb`
- Stage 0 train candidate:
  `exp397_prefix_gr_agreement_adaptive_sigma_exact_hmm_compact_selfcontained_train.ipynb`
- inference notebook scaffold:
  `exp397_prefix_gr_agreement_adaptive_sigma_exact_hmm_inference.ipynb`
- 正規train notebookへcompact self-contained Stage 0を採用済み。
- Kaggle private CPU version 1は完了。Stage 1、inference、submissionは未実装・未実行。
- 詳細設計:
  `docs/legacy/steering/20260725-exp397-prefix-gr-agreement-adaptive-sigma-exact-hmm/`

## 結果

| メトリック | 値 |
| --- | --- |
| Stage 0 gate | FAIL（4/7 PASS） |
| poor multiplier fraction | `8/773 = 0.0103493` |
| full/tail multiplier agreement | `0.666235` |
| full/tail Spearman | `0.167466` |
| runtime | `39.3598 sec` |
| CV | - |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- prefix agreementとbase residual scaleを別の量として定義し、変更をbounded softeningだけに限定した。
- exp307 / exp346のscale縮小を再開せず、exp209のno-op経路を明示した。
- horizontal truthを読まないreader、agreement / coefficient SHA freeze、7条件AND gateを
  self-contained candidate上に実装し、専用11 testsとstrict experiment validationがPASSした。
- 773 wellsすべてでfull/tail agreementを評価でき、freeze前truth読込0、HMM/model/booster 0で
  selectorの適格性を低コストに判定できた。

### 悪かった点

- discussionの `1.3` 改善はscore、CV、versionの詳細がなく、HMMへの転用は未検証。
- Pearson相関は絶対biasや局所的なずれを捉えない。
- poor groupは8 wellsだけで非退化gateをFAILし、full/tailの係数一致と順位相関もFAILした。

### リスク / 注意

- threshold、multiplier、support、windowは結果後に調整しない。
- Stage 0 FAILを受けてStage 1、inference、submissionを起動しない。
- gate、threshold、multiplier、support、window、相関種の事後調整を行わない。

## 次

- `stage_0_failed_close_without_rescue`としてbranchを閉じる。
- version 2や同family rescue backlogを追加しない。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
