# タスクリスト

## 設計で完了

- exp299をexp223 parent / exp296 negative reference / exp209 base parity referenceとして切り出した。
- Pass A Type Well-only exact HMMからPass Bへの一方向controller依存を固定した。
- outside candidate exact 0、base mean outside/boundary rowのall-state exact 0を固定した。
- exp223 sigma 12 ftを使うboundary fadeと、base posterior inside massを使うrow gateを固定した。
- support内のbase-posterior-weighted likelihood massを保存するconditional normalizationを固定した。
- scientific variant 1、Pass A/B合計1,546 HMM well-runs、LightGBM config / trained fold / booster 0、control再実行0を固定した。
- technical/performance/fold/scope/hidden-like/worst-well gateとFAIL-closeを固定した。
- 再現性、truth-late freeze、SHA、Kaggle CPU/runtime境界を固定した。
- steering、実験scaffold、backlogを作成した。

## 実装で完了

- exp296 compact sourceを構成基準に、exp223/exp209 Type Well HMMとexp223 self-GR関数だけを持つ別名Jupytext compact self-contained train候補を作成した。
- Pass A posterior/mean SHA freeze、exp209 float32保存契約parity、row gate、conditional normalization、Pass Bを実装した。
- synthetic contract testsでoutside zero、boundary/all-state zero、mass preservation、signed inside contribution、non-circular dependencyを確認した。
- truth/control late join、fold/scope/upper-boundary/hidden-like/by-well/step metricsとSHA manifestを実装した。
- fail-closed inference候補を追加し、raw test読み込み・prediction・submission生成を実装していない。
- py_compile、Ruff、Jupytext round-trip、専用12 tests、experiment/template validationをPASSした。repository testsは354 passed / 1 skippedで、今回未変更のexp296既知2件だけがFAILした。
- compact候補と親compactの章立て・記載量を比較した。exp296 `2,260`行・10章に対してexp299 `2,758`行・10章で、正規Notebookは別承認まで変更していない。

## 2026-07-20 実行承認

- ユーザーの「実行してください」により承認された正規train Notebook採用とKaggle private CPU push 1回は、version 1（id_no `127957958`）で消費済み。v1は全1,546 HMM well-runs後、truth/control readout前のexp209 float32 CSV parity bugでERRORとなった。
- parity修正後の再度の「実行してください」により、同じcanonical slugのversion 2を1回だけ再実行する承認を得て消費した。
- 承認範囲は1 scientific variant、Pass A/B合計1,546 HMM well-runs、0 booster、control再実行0。
- inference実装、raw-test生成、submissionは未承認。

## 実行前count確認

- scientific variant: 1。
- internal HMM passes: Pass A base-only 1 + Pass B handoff 1。
- wells: 773。
- total HMM well-runs: `773 + 773 = 1,546`。
- LightGBM config / trained fold / booster: `0 / 0 / 0`。
- parent exp223 / exp296 / exp209 control retraining: 0。
- GPU: 0。

## 次のアクション

version 2はexp209 parity exact 0で完了したが、candidate RMSE `11.789577561`、exp223比`+0.439634615 ft`、改善0/5 folds、performance 2/11 PASSでFAILした。事前固定fail actionどおりbranchを閉じ、version 3 repush、inference、submissionへ進めない。

## 完了

- `kaggle-review-exp`、`kaggle-strategy`、`docs/06_reproducibility.md`を確認した。
- exp296のoutside rowsが全てupper側であり、state-wise hard maskが相対boundary priorになった証拠を反映した。
- 実験番号exp299の空きを確認した。
- `docs/legacy/steering/20260720-exp299-base-posterior-self-gr-boundary-handoff/`を作成した。
- `experiments/exp299_base_posterior_self_gr_boundary_handoff/`をtemplateから作成した。
- JSON/YAML contract、strict experiment validation、project template validationを通した。
- `kaggle-review-exp` reviewerでcore evidence categoryが対象文書群に揃っていることを確認した。
- 設計時点では`experiment_summary.md`を295 experimentsへ更新し、exp223からexp299へのlineageと`design_locked_not_implemented`を反映した。実装後は同時点で存在する296 experimentsへ再生成し、exp299を`implemented_waiting_for_notebook_adoption`へ更新した。
- Kaggle private CPU version 2（id_no `127957958`）をPass A/B各773 wells、合計1,546 HMM well-runs、0 boosterで完了し、完了ログからCV、25 technical gates、11 performance gates、生成物path、SHAを記録した。
- technicalはrow gate maxの`2.9e-15`超過だけがFAIL、performanceは9/11 FAILであり、negative resultとしてbranchを閉じた。
