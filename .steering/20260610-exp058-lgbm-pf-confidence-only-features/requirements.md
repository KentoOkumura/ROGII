# 要件

## 依頼

`lgbm_pf_confidence_only_features` を実装する。`exp051/052` の LightGBM
capacity pseudo-tail surface に PF/Beam の信頼度・不一致診断だけを追加し、
same-run geometry control と比較できる train-side audit を作る。

## 制約

- Route: `ml_model`
- PF raw prediction、Beam raw prediction、public PF gate、hidden branch replacement は特徴量に入れない。
- `exp051` の LightGBM capacity params、pseudo-tail cutoff policy、distance-balanced sampling、row cap、residual shrink、fixed bucket-shrink postprocess を固定する。
- 初回の notebook 実行は Kaggle 上で行う。ローカル notebook 実行はしない。
- 現在の exp029 artifact に `PF-exp052 diff` がない場合は、`abs_pf_pred_minus_exp026_oof` を proxy として明記する。

## 受け入れ基準

- `.steering/20260610-exp058-lgbm-pf-confidence-only-features/` が存在し、要件・設計・タスクが記入されている。
- `experiments/exp058_lgbm_pf_confidence_only_features/` に config、train / inference notebook、audit script、記録ファイルがある。
- LightGBM geometry control と PF/Beam confidence-only variant が定義されている。
- supported 判定は confidence-only candidate が paired geometry control を original-fold / well-hash の両方で上回ること。
- 出力 metrics / summary は `pf_confidence_*` 名で保存される。
- `scripts/validate_experiment.py --experiment exp058_lgbm_pf_confidence_only_features` が通る。
