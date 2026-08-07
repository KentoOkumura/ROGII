# 要件

## 依頼

`prefix_crop_window_features_on_exp148` を実装する。実行は CPU 前提にする。

## 制約

- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- exp148 control は再学習しない。保存済み exp148 CV / Public LB を比較基準にする。
- 既存 exp148 feature は置換せず、crop-window features を add-only する。
- PF/Beam candidate generation、U-projection、learned probability / expected error は crop-window 版へ置き換えない。
- 学習・評価行は削除しない。
- crop 境界は hidden current-test で再現できる target-free 条件だけを使う。
- Kaggle runtime は CPU (`enable_gpu=false`) にする。

## 受け入れ基準

- `experiments/exp161_prefix_crop_window_features_on_exp148/` が作成されている。
- `config.yaml` に route、親実験、CPU mode、active variant、15 boosters 予定が記録されている。
- `prefix_crop_window` feature group が train / inference の両方で同じ schema になる。
- train / inference notebook が exp161 名で存在する。
- py_compile、ruff F821、notebook JSON validation、`validate-exp` が通る。
