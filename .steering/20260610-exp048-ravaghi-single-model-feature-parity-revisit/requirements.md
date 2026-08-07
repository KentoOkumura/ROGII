# 要件

## 依頼

`ravaghi_single_model_feature_parity_revisit` を `exp048` として実装する。

## 制約

- Route: `ml_model`
- 親実験は `exp043_ravaghi_feature_family_ablation_matrix` とする。
- 入力は `exp029_public_sel15_pf_oof_feature_generation` の train well の途中以降を隠した疑似 test 生成物を使う。
- direct PF/Beam replacement、Ridge/meta-stack、learned router、train-only formation columns は使わない。
- Ravaghi exact beam / NCC / GR match は、validation target を使わず cutoff 以降の `TVT_input` を隠した状態で再生成する。
- raw、fixed bucket shrink、anchor gate、public PF blend を別候補として記録する。

## 受け入れ基準

- `experiments/exp048_ravaghi_single_model_feature_parity_revisit/` が作成されている。
- `config.yaml` の route、親実験、比較基準、postprocess 候補が `exp048` の目的に合っている。
- `ravaghi_single_lgbm_audit.py` が `single_lgbm_feature_parity_report.csv` を保存する。
- supported candidate は `base_geometry_bucket_shrink`、`public_pf_selector`、`pf090_hold010` を original-fold / well-hash の両方で上回る場合だけ選ばれる。
- `validate_experiment.py`、ruff、py_compile、local smoke が通る。
