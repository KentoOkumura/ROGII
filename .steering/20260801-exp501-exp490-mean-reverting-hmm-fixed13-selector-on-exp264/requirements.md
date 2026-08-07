# 要件

## 依頼

保存済み `exp490_geometry_centered_mean_reverting_offset_hmm` の full OOF prediction を、
修正版 `exp264_exp263_candidate_confidence_dual_selector` の固定12候補に13番目の
primary candidateとして追加する実験を設計する。今回はバックログ、steering、実験scaffold、
設定・記録文書だけを作成し、selectorコード、contract test、Notebook実装、Kaggle実行は行わない。

## 目的

exp490はfull OOF RMSE `8.480155260 ft`で保存exp357親を`1.257039898 ft`、
exp226 finalを`0.946954337 ft`改善し、4/5 foldsとpersistent episode SSE
`41.409965%`削減を得た。一方、by-well p95は`+7.257813771 ft`、worstは
`+49.602560348 ft`であり、固定候補としてはfail-closedである。

本実験は、exp490を単独採用またはwell単位でroutingするのではなく、exp264の
candidate-long dual-objective selectorへ13番目の候補として追加し、target-freeな
行・候補文脈から安全な局所だけを選べるかを検証する設計を固定する。

## 制約

- 対象実験: `exp501_exp490_mean_reverting_hmm_fixed13_selector_on_exp264`
- Route: `ensemble`
- selector親: `exp264_exp263_candidate_confidence_dual_selector`
- 追加候補親: `exp490_geometry_centered_mean_reverting_offset_hmm`
- 実装参照: `exp496_exp486_absolute_geometry_fixed13_selector_on_exp264`
- 変更する変数はprimary candidate bankへの`geometry_mean_reverting_hmm` 1件追加と、
  その候補に固有なtarget-free confidence 2列の追加だけとする。
- 親12候補、7候補fixed fallback、2 objectives、outer 5 × inner 4 split、sampling、
  LightGBM設定、score式、閾値、評価scopeを変更しない。
- exp490 / exp226 / exp357 predictionは保存生成物だけを使い、HMM、PF、Beam、親selector、
  downstream TVT controlを再実行・再学習しない。
- exp498のphysics well featuresとexp499のwell router/model/featureは使用しない。
- exp490の`fold`、truth、error、role、episode、by-well outcome、gate結果はfeature freeze前に
  読まない。exp264 outer foldへglobal key join後に再partitionする。
- exp490 Stage 1のfail-closeとexp499 router FAILを再分類しない。
- 既存fixed13実験の結果を見てthreshold、weight、feature、candidate subset、domain、gateを
  調整しない。
- 今回はdesign-onlyであり、実装、正規Notebook採用、Kaggle package/push/run、current-test、
  downstream TVT、inference、submissionを承認しない。
- 再現性は`docs/06_reproducibility.md`に従い、gzipはraw SHAとdecompressed content SHAを
  分け、後者を主証拠にする。

## 固定実行契約

- scientific variant: 1
- selector objectives: 2（`pred_abs_error`、`p_within10`）
- reporting outer folds: 5
- inner folds per outer fold: 4
- 実装・実行を別承認した場合のselector booster: `1 × 2 × 5 × 4 = 40`
- expected compact partitions: 25
- expected compact rows: 18,919,945
- expected outer-valid candidate-long rows: 49,191,857
- parent/control retraining: 0
- HMM / PF / Beam well-runs: 0 / 0 / 0
- GPU booster: 0
- downstream TVT booster: 0

## 受け入れ基準

- `KAGGLE_DIRECTION.md`の未着手バックログへ、優先度、根拠、検証方法、禁止事項を含む
  exp501項目が追加されている。
- `.steering/20260801-exp501-exp490-mean-reverting-hmm-fixed13-selector-on-exp264/`に
  要件、設計、タスクリストがあり、未記入項目が残っていない。
- `experiments/exp501_exp490_mean_reverting_hmm_fixed13_selector_on_exp264/`に
  design-onlyの`config.yaml`、`README.md`、`SESSION_NOTES.md`、`result.md`、
  `metrics.json`と未実装placeholder Notebookがある。
- `config.yaml`に`experiment.route: ensemble`、lineage、入力ファイル/SHA、候補契約、
  feature freeze、40 boosterの実行量、technical/score/integration gate、禁止事項がある。
- exp490入力は3,783,989 rows / 773 wells、prediction raw gzip SHA
  `99030b33...61b72c`、decompressed SHA `e020e82e...09a07`を固定する。
- feature freeze前の追加候補allowlistはkey列、`geometry_mean_reverting_hmm`、
  `geometry_mean_reverting_delta_mean`、`geometry_mean_reverting_hmm_std`に限定する。
- 成功判定はtechnical、leakage、dual score、候補利用、pooled/fold/scope、by-well tailの
  全ANDで事前登録され、FAIL時はsame-OOF rescueなしで閉じる。
- design-only時点ではdeterministic anchor、CV改善、提出候補を主張しない。

## 次のアクション

この受け入れ基準を満たした後は設計確定で停止する。ユーザーが別途実装を承認した場合だけ、
exp496 compact self-contained trainを構成参照にして別名Jupytext候補とcontract testを作る。
