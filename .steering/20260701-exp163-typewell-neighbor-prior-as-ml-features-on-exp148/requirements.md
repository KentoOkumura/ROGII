# 要件

## 依頼

`typewell_neighbor_prior_as_ml_features_on_exp148` を実装する。CPU 前提とし、Kaggle timeout 対策として LightGBM config ごとに `lgb0` / `lgb1` / `lgb2` の train notebook を分ける。

## 制約

- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- Control 再学習はしない。保存済み exp148 CV / Public LB を比較基準にする。
- `runtime.kaggle.enable_gpu=false` とする。
- `lgb0` / `lgb1` / `lgb2` は別 notebook / 別 Kaggle kernel として準備できること。
- typewell neighbor prior は fold-safe に作る。validation well と same-fold validation true TVT を neighbor pool に入れない。
- Prior TVT は direct replacement / hard selector / postprocess として使わず、add-only confidence feature に変換する。

## 受け入れ基準

- `.steering/`、`experiments/exp163_typewell_neighbor_prior_as_ml_features_on_exp148/`、`config.yaml`、SESSION_NOTES / result / metrics が exp163 として整っている。
- `train_lgb0.py`、`train_lgb1.py`、`train_lgb2.py` と対応 `.ipynb` が存在する。
- Kaggle package 用 metadata で CPU / internet off / required source kernels が設定されている。
- `py_compile`、`ruff --select F821`、`make validate-exp` が通る。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録する。
