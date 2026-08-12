# 要件

## 依頼

`xgb_catboost_pf_confidence_only_features` を実装する。XGBoost / CatBoost の
pseudo-tail residual 系候補に、public PF/Beam の予測値そのものではなく
信頼度・不一致だけを特徴量として追加し、geometry control と比較できる状態にする。

## 制約

- Route: `ml_model`
- PF prediction、Beam prediction、public PF gate、hidden branch replacement、直接候補選択は使わない。
- 追加特徴は PF seed spread、likelihood / particle diagnostics、beam spread、PF/Beam disagreement、PF/anchor disagreement、利用可能な PF/model disagreement に限定する。
- 現在の exp029 artifact には `PF-exp052 diff` がないため、利用可能な `PF-exp026 OOF diff` を proxy として使い、その差分を config / notes に記録する。
- train well の途中以降を隠した exp029 OOF 風 artifact を使い、validation well の target や後続 TVT_input を特徴生成に混ぜない。
- 推論 port / submit はこの実装範囲に含めない。

## 受け入れ基準

- `docs/legacy/steering/20260610-exp057-xgb-catboost-pf-confidence-only-features/` が存在し、要件・設計・タスクが記入されている。
- `experiments/exp057_xgb_catboost_pf_confidence_only_features/` に config、train / inference notebook、audit script、記録ファイルがある。
- XGBoost と CatBoost それぞれに geometry control と PF confidence-only variant が定義されている。
- audit script が variant ごとに estimator / params を切り替えられる。
- 出力 metrics / summary は `pf_confidence_*` 名で保存される。
- `scripts/validate_experiment.py --experiment exp057_xgb_catboost_pf_confidence_only_features` が通る。
