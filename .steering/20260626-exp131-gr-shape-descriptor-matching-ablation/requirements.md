# 要件

## 依頼

`KAGGLE_DIRECTION.md` の backlog `gr_shape_descriptor_matching_ablation` を実装する。raw GR/NCC/DTW を直接 TVT 候補や LightGBM 特徴にするのではなく、local shape descriptor matching cost が既存 PF/Beam 候補の良し悪しを target-free に説明できるかを評価する。

## 制約

- Route: `pf_beam`
- 親: `gr_shape_descriptor_matching_ablation` backlog、比較文脈は `exp099` / `exp112` / `exp128`
- 入力候補 surface は exp072 deterministic full replay train cache に固定する。
- 対象候補は `pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb`。
- validation true TVT を window center、descriptor normalization、candidate generation に使わない。
- shuffled-GR / no-GR negative control を必ず実装する。
- direct TVT candidate、ML add-only 大量投入、inference port、submission はこの実験では作らない。
- 再現性: `docs/06_reproducibility.md` に従い、input SHA、gzip decompressed SHA、score top1 proxy SHA を記録する。

## 受け入れ基準

- `exp131_gr_shape_descriptor_matching_ablation` フォルダに config、settings、train/inference notebook、補助 `.py`、README、SESSION_NOTES、result、metrics がある。
- train notebook 上で目的、設定確認、入力契約、監査実行、出力 preview、metrics 保存が追える。
- 補助 `.py` が raw point、NCC、banded local shift、shape descriptor、combo descriptor、shuffled/no-GR controls を生成できる。
- candidate AUC/logloss、topK coverage、rank-score top1、bucket / by-well stress を保存する。
- `validate_experiment.py` が通る。
- deterministic anchor として扱わないことが config / notes に明記されている。
