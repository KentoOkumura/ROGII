# 要件

## 依頼

`z_driven_pf_z_candidate_gate` を実装する。

## 制約

- Route: `pf_beam`
- 親実験は `exp072_exp063_full_replay_feature_cache` とし、保存済み train pseudo-tail feature cache を固定入力にする。
- default 予測は `likpf_mean` に固定し、`pf_z` direct replacement はしない。
- `pf_z` への switch rate は 1-10% 程度の低頻度 cap を持たせる。
- gate 条件には valid/test true TVT、oracle error、true-error rank を使わない。
- gate 条件は raw-test-compatible な `dzdmd`、候補差分、PF/Beam disagreement、candidate path slope/curvature、`md_since`、tail rank、near-prefix guard に限定する。
- row-wise switch だけでなく minimum segment length 付き segment gate を含める。
- 新規モデル学習、PF particle 再生成、提出ファイル生成は行わない。
- 再現性は `docs/06_reproducibility.md` に従い、gzip 入力は decompressed content SHA を記録する。

## 受け入れ基準

- `docs/legacy/steering/`、`config.yaml`、train/inference notebook、補助 `.py`、`SESSION_NOTES.md`、`result.md` が更新されている。
- train notebook から設定確認、入力確認、gate grid、audit 実行、生成物確認が追える。
- `metrics.csv`、`gate_variants.csv`、`by_well.csv`、`bucket_metrics.csv`、`representative_wells.csv`、`rawtest_parity_checklist.csv`、`summary.json` を生成する実装がある。
- 静的検証として py_compile、notebook JSON 検証、`validate-exp`、ruff check/format check が通る。
- Kaggle train 実行は別アクションとして残し、ローカル notebook 実行は行わない。
