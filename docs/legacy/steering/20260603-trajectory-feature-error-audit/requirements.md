# 要件

## 依頼

`KAGGLE_DIRECTION.md` のアイデアバックログ先頭にある
`trajectory_feature_error_audit` を実装する。

`exp010_trajectory_drift_ablation` で trajectory feature が CV を悪化させたため、
次の feature / router 実験へ進む前に well-level の悪化条件を診断する。

## 制約

- 新規提出候補ではない。Kaggle Notebook push / submission は行わない。
- 入力は `experiments/exp010_trajectory_drift_ablation/artifacts/well_metrics.csv` を正とする。
- GR missing と hard-well tag は
  `experiments/exp006_hard_well_router_diagnostic/artifacts/router_diagnostic_well_tags.csv`
  から結合する。
- public-like 3 wells だけに合わせた判断はしない。
- train-only formation columns や隠し test 情報は使わない。

## 受け入れ基準

- `control_exp003_no_gr` と trajectory variants の well-level RMSE 差分を出力する。
- eval length、trajectory slope、GR missing、fold、hard-well tag で集計できる。
- top hurt / top help wells を確認できる。
- 診断結果を CSV / JSON / Markdown として artifacts に保存する。
- 結果を `SESSION_NOTES.md`、`result.md`、`KAGGLE_DIRECTION.md` に反映する。
