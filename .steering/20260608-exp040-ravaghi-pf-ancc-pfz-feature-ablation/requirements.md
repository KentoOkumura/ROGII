# 要件

## 依頼

`ravaghi_pf_ancc_pfz_feature_ablation` を `exp040` として実装する。

## 制約

- Route: `ml_model`
- 親実験は `exp026_pseudo_tail_bucket_shrink_inference_submit` とし、入力 artifact は `exp029_public_sel15_pf_oof_feature_generation` を使う。
- train-only の `ANCC`、`ASTNU`、`ASTNL`、`EGFDU`、`EGFDL`、`BUDA` は直接 feature として読まない。
- `pf_error`、`last_anchor_error`、`beam_error`、`exp026_oof`、exp026 bridge columns は model feature にしない。
- PF 候補値を直接 replacement / stack として採用せず、PF delta、PF uncertainty、PF-vs-Z disagreement、likelihood margin の feature ablation として扱う。
- 初回 full CV は Kaggle Notebook 実行を正とし、ローカル notebook 実行はしない。

## 受け入れ基準

- `.steering/`、`config.yaml`、train/inference notebook、監査スクリプト、`SESSION_NOTES.md`、`result.md`、`metrics.json` が `exp040` として整っている。
- `model.variants` が `base_geometry` と PF ANCC/PFZ proxy family を比較できる。
- static validation と local script smoke が通る。
- full audit 後は original-fold と well-hash の両方で `base_geometry_bucket_shrink` を上回る場合だけ supported candidate と記録する。
