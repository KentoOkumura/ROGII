# exp496_exp486_absolute_geometry_fixed13_selector_on_exp264 セッションノート

## 目的

exp486 Absolute版の保存済みtrain predictionをcorrected exp264 fixed12 selectorへ
13本目として追加し、単独では危険なpooled改善をselectorが安全に局所利用できるか
検証するStage A/C実装を、凍結設計どおりに準備する。

## 現在の状態

- Route: `ensemble`
- status: `completed_train_side_rejected_no_submit`
- implementation: ユーザー承認済み・compact候補実装完了
- Kaggle Stage A/C run: CPU version 1 `COMPLETE`
- inference / submission: 未承認・未実行
- canonical train Notebook: compact候補を採用
- canonical inference Notebook: template placeholder
- compact train / inference guard: Jupytext `.py` / `.ipynb`作成済み
- CV: fixed13 hard `8.461357622`、parent fixed12比`-0.191174334 ft`
- scientific decision: `FAIL_CLOSE_EXP486_ABSOLUTE_FIXED13_SELECTOR`
- LB: なし

## 設計根拠

exp486 Stage 1:

- Absolute RMSE: `9.726938029`
- saved exp404 control: `10.914522073`
- pooled gain: `+1.187584044 ft`
- improved folds: `4 / 5`
- by-well delta p95 / worst: `+10.069321492 / +44.021977054 ft`
- Residual RMSE / gain: `11.139812021 / -0.225289948 ft`

Absoluteは平均と全事前scopeを改善したがtail gateをFAILした。Residualはpooledでも
悪化したため、1変更1仮説を守ってAbsoluteだけを候補化する。exp392などfixed13
実験では追加候補非利用wellでもincumbent rerankingがtailを壊したため、parent
fixed12 scoreとのpaired監査とreranking診断を設計に含めた。

## 固定契約

- score candidates: 13
- primary hard-select candidates: 12
- unchanged fixed fallback candidates: 7
- active variant: 1
- objectives: 2
- outer / inner folds: `5 / 4`
- planned CPU selector boosters: `40`
- expected compact partitions / rows: `25 / 18,919,945`
- expected outer-valid candidate-score rows: `49,191,857`
- parent/control retraining: 0
- candidate PF / HMM / Beam rerun: `0 / 0 / 0`
- GPU / downstream TVT / inference / submission: `0 / 0 / 0 / 0`

## exp486入力契約

- source dataset: `kentookumura/exp486-v2-stage1-frozen-targetfree`
- source kernel: `kentookumura/exp486-exp226-geometry-residual-likpf-train` v4
- scientific contract SHA:
  `62dcb499c0c9c9320091fa28663771493847dd6f46f03737015d1373dddc5f8e`
- prediction raw / decompressed / upstream logical SHA:
  `0fe0cdda02c49eaa80ab668cb8e68e5b3e02b98f46a5105be6818497f2b65de3` /
  `05f692238c53711172f5e4e430eb46766cd26f2e3dac92472cb211b5639153e6` /
  `70a5ac662c9c58fe54d050f1350ed08e912ecb4edc6362e98e3c3663cd704ea8`
- absolute ledger raw / decompressed SHA:
  `3f7381f1265d9b5bc9f0b9a68d4a3a088a620bfc04d674faca0d608390ea7b96` /
  `ef76d89fef9529d11501a5c17999e95e22d45006c0fd1e6b80e71d674b6c5a80`
- exp226 geometry decompressed SHA:
  `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`

predictionは`id,well_id,row_idx,suffix_offset,absolute prediction`だけ、absolute
ledgerは5つのtarget-free confidenceだけを許可する。Residual、truth、control、
fold/role、scope、gate、by-well結果はfeature freeze前に開かない。

## 科学gate

- selector score 3指標がpooledと4/5 folds以上でpriorを改善
- exp486 top1率`>=0.5%`、positive usage folds`>=4`
- fixed13 pooledがparent fixed12以下、改善fold`>=4/5`
- raw observed/missing、高missing、near、1000+、hidden-like 2面のdelta各`<=+0.02 ft`
- by-well p95 / worst delta各`<=+0.25 ft`

全項目AND。FAIL時は`FAIL_CLOSE_EXP486_ABSOLUTE_FIXED13_SELECTOR`。同じOOFで
weight、threshold、domain、feature、gate、Residual、blendを救済しない。

## 再現性

- `docs/06_reproducibility.md`確認済み
- seed 42、stable SHA256 candidate-long sampling
- LightGBM `deterministic=true`, `force_col_wise=true`
- parallel fit前にsamplingを固定し、worker内global RNGを使わない
- 保存exp486 predictionを使い、PF乱数生成は0
- input、feature schema/content、40-model manifest、candidate score、compact、gate、
  summary、Kaggle version SHAを記録済み
- stochastic selectorかつsubmissionなしのためdeterministic submission anchorとはしない

## 作成ログ

- 2026-07-31: `task new-steering`は環境に`task`がなく実行不可。
- 2026-07-31: 同等手順`make new-steering`でsteeringを作成。
- 2026-07-31: `make new-exp`でdesign-only experiment scaffoldを作成。
- 2026-07-31: steering、config、candidate / feature / output contract、README、
  SESSION_NOTES、result、metrics、backlogを設計内容へ更新。
- 2026-07-31: template Notebookの汎用metrics/submission実行セルを除去し、
  train / inferenceをmarkdown-only design placeholderへ戻した。
- 2026-07-31: ユーザーの「exp496を実装してください」で実装承認を受けた。
- 2026-07-31: `src/exp486_fixed13_candidate_cache.py`を作成し、prediction、
  absolute ledger、freeze manifestのraw/decompressed/logical/scientific contract
  SHA監査、厳密allowlist、global key join、5 native confidenceを実装した。
- 2026-07-31: 9章構成のcompact self-contained train候補を作成した。Stage A
  schema freeze、outer 5 × inner 4のStage C、parent fixed12 paired gate、raw GR
  observed/missing・高missing・0--250・1000+・hidden-like 2面、by-well tail、
  H512/whole-well oracle、score margin/entropy reranking、feature importance、
  reproducibility summaryをNotebook上で追えるようにした。
- 2026-07-31: current-test exp486候補、downstream TVT、inference、submissionを
  明示的に停止するcompact inference guardを作成した。
- 2026-07-31: 専用testを追加し、`10 passed`。exp264 pipeline、exp392 fixed13、
  notebook testを含む関連回帰は`41 passed`。
- 2026-07-31: Jupytext roundtrip、`py_compile`、`ruff` F821/all、
  `make validate-exp EXP=exp496_exp486_absolute_geometry_fixed13_selector_on_exp264`
  をPASSした。
- 2026-07-31: 親compactのexp392は8章 / 540行、exp496は9章 / 647行。
  exp486二入力監査、reranking、feature importance、再現性summaryを追加し、
  親より章・記載量が欠けていないことを確認した。
- canonical Notebookは既存placeholderを上書きせず、Kaggle package、Kaggle API、
  学習、推論、提出は実行していない。
- 2026-07-31: ユーザーの「実行してください」で、正規train Notebook採用と
  Kaggle Stage A/C runの承認を受けた。承認scopeは`1 variant / 2 objectives /
  outer 5 / inner 4 / 40 CPU selector boosters / parent-control retraining 0 /
  candidate PF-HMM-Beam rerun 0 / GPU 0 / downstream TVT 0 / inference 0 /
  submission 0`。保存済み親結果だけを比較対象に使い、controlを再学習しない。
- 2026-07-31: compact train `.ipynb`を正規train Notebookへ採用し、canonical
  kernel候補`kentookumura/exp496-exp486-absolute-geometry-fixed13-selector-train`を
  `run_on_push=true`、CPU、internet/GPU/TPU falseでstrict package化した。
  dataset sourceはexp486 frozen target-free、kernel sourceはexp263/264だけ。
  bootstrap zipは36 filesで、selector pipeline、exp486 fixed13 helper、exp251 schema、
  hidden-like assignmentを含むことを確認した。package notebook / metadata SHAは
  `2c9cd7784fdf205c706b92c06d632a072a18547e25dbebb7ec4b5b4274224997` /
  `b2d58f186500ba7d5ab350df66b8127817df3e6bd46689e05e4c1b563d927604`。
- 2026-07-31: 上記54文字slugは事前pull 403、list `Not found`の後、初回pushが
  詳細なし`SaveKernel 400 Bad Request`で終了した。Kaggle実行・booster fitは開始
  していない。id/title slugは一致していたため、exp264などの既知の長slug制約と
  同じと判断した。同じexp496、Notebook、科学契約のまま、意味を保った短い
  canonical id/title `kentookumura/exp496-exp486-fixed13-selector-train` /
  `exp496 exp486 fixed13 selector train`へ同時に揃える。元slugは再pushしない。
- 2026-07-31: 短縮canonical packageを再生成し、id/title slug一致、private、
  `run_on_push=true`、CPU、GPU/TPU/internet false、exp486 dataset source、exp263/264
  kernel sourceを再確認した。最終package notebook / metadata SHAは
  `714546ac692bcabd0469faaea16cced042412e71417c19af4e2c1abbc67db284` /
  `4214bb82bccfc7bf026f4d525cb04e4a53bc7bbb3a90260ad60f26a11890ce45`。
- 2026-07-31: 短縮slugの事前pullは403で未作成。canonical packageをpushし、
  Kaggle CPU version 1（id_no `129287597`）の作成と`RUNNING`を確認した。
  push後metadataはprivate、CPU、GPU/TPU/internet false、exp486 dataset source、
  exp263/264 kernel source、competition sourceを全て照合した。同じversionを再pushせず
  terminal statusまで監視する。
- 2026-07-31: version 1は`3945.563 sec`で`COMPLETE`。40/40 selector models、
  25 compact partitions / 18,919,945 rows、49,191,857 outer-valid score rows、
  technical checks 10/10、leakage audit、selector score guardを全PASSした。
  Stage Aは158候補featureから95列を選び、compact metaは固定77列。
- 2026-07-31: fixed13 hard OOFはparent fixed12 `8.652531955610227`から
  `8.461357621859406`へ`0.191174333751 ft`改善。fold 0/1/2/4を改善し`4/5`、
  exp486 top1は420,211 rows / `11.104974%`、全5 foldsで正。raw observed/missing、
  high-missing、0--250、1000+、hidden-like 2面の固定7 scopeを全PASSした。
- 2026-07-31: by-wellは416改善 / 357悪化、delta p95 `+1.109359862 ft`、
  worst `14fee784 +9.361781278 ft`で、固定`+0.25 ft`上限を両方FAIL。
  decision=`FAIL_CLOSE_EXP486_ABSOLUTE_FIXED13_SELECTOR`。pooled / fold / scope改善で
  tail FAILを救済せず、downstream TVT、current-test、inference、submissionへ進まない。
- 2026-07-31: post-freeze診断はH512 / whole-well oracle headroom
  `0.097475 / 0.066659 ft`、exp486非top1行のincumbent change率`34.789662%`、
  usage-delta Pearson / Spearman `0.020819 / -0.000958`。利用0の38 wellsでも
  24改善 / 14悪化で、direct usageだけではtail差を説明できない。
- 2026-07-31: 完全logs（SHA
  `40a731b6cf3c4e42c7d752be93ee85e28ff5ba60e3b0891a012820d15d19eb44`）と、
  scope / usage / by-well / gate / summary / feature importance / model / compact /
  reproducibility manifestの小さい生成物だけを`kaggle/output/train_v1/`へ選択取得。
  巨大score parquetと25 compact partitionsは取得していない。model / compact /
  score SHAは`d4ac1528...36a977` / `7bacee14...6b2ac` / `07588f6d...5003e`。

## 次のアクション

branchをterminal closeする。同じOOFでweight / threshold / domain / feature / gateを
調整しない。追加原因確認が必要な場合だけ、既存backlogの0-booster
`fixed13_selector_incumbent_reranking_instability_readout`へexp496保存scoreを加える。
