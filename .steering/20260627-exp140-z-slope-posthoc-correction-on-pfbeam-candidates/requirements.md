# 要件

## 依頼

`z_slope_posthoc_correction_on_pfbeam_candidates` を実装する。

## 制約

- Route: `pf_beam`
- 親: `exp072_exp063_full_replay_feature_cache`
- 既存 PF/Beam 候補を固定し、PF/Beam 本体の再生成や supervised selector 学習はしない。
- `dZ/dMD` は target-free な raw horizontal covariate から作り、true TVT は scoring のみに使う。
- 全区間補正は禁止。near prefix guard、Z slope gate、候補 disagreement、`pf_z` direction/pull の低頻度補助を比較する。
- 補正量は ±10ft / ±20ft 程度で clip する。
- 再現性は `docs/06_reproducibility.md` に従い、入力 cache SHA と gzip decompressed SHA を記録する。
- Kaggle Notebook 実行を正とし、ローカル notebook 実行はしない。

## 受け入れ基準

- `.steering/20260627-exp140-z-slope-posthoc-correction-on-pfbeam-candidates/` に requirements / design / tasklist がある。
- `experiments/exp140_z_slope_posthoc_correction_on_pfbeam_candidates/` に config、settings、train/inference notebook、補助スクリプトがある。
- train notebook は設定確認、入力契約、監査実行、出力 preview、metrics/SHA 表示をセルで追える。
- `config.yaml` の `experiment.route` は `pf_beam`。
- 生成物として candidate metrics、bucket metrics、by-well、group metrics、上位 variant OOF gzip、feature schema、summary JSON を保存する。
- inference notebook は no-op とし、train-side diagnostic であることを明記する。
- `py_compile`、`ruff check`、`make validate-exp` が通る。
