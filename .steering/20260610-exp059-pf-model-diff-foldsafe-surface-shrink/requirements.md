# 要件

## 依頼

`pf_model_diff_foldsafe_surface_shrink` を実装する。

## 制約

- Route: `ml_model`
- 親 surface は `exp058_lgbm_pf_confidence_only_features` と同じ exp029 pseudo-test rows に固定する。
- `exp052/054` 相当の予測は、validation fold 自身を使わず train-fold wells だけで再生成した fold-out 予測に限定する。
- PF / Beam の raw prediction は直接置き換え候補にせず、fold-out ML anchor との差分・信頼度特徴として使う。
- bucket shrink alpha は same-OOF fit を避け、各 held-out split に対して他 split だけで推定する。
- exp014 の固定 alpha は再利用しない。

## 受け入れ基準

- `exp059_pf_model_diff_foldsafe_surface_shrink` が作成され、train notebook が新しい audit script を参照する。
- config に `ml_model` route、親実験、fold-safe leakage policy、fold-out source、fold-out shrink 候補が明記される。
- train audit script が exp052/054 fold-out source prediction、PF/Beam-vs-model diff features、fold-out bucket shrink alpha summary を生成できる。
- `scripts/validate_experiment.py` と Python compile / ruff が通る。
