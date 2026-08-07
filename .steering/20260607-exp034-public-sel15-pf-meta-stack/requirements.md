# 要件

## 依頼

`public_sel15_pf_meta_stack` を実装する。

## 制約

- Route: `pf_beam`
- 親 artifact は `exp029_public_sel15_pf_oof_feature_generation` の public sel15 PF/Beam OOF-like rows。
- self-route anchor は `exp026_pseudo_tail_bucket_shrink_inference_submit` の clean CV 12.870780 として扱う。
- exp029 artifact の `exp026_oof` は全欠損なので、そのまま特徴量に使わず exp026-style pseudo-tail prediction を fold-safe に再生成する。
- 見えない test well 用の PF prediction だけで学習しない。
- Public LB 8.x の replay に合わせて最適化せず、まず clean train well の途中以降を隠した疑似 test CV で判断する。
- 実験は audit-only とし、supported candidate が出るまで inference port / submission は作らない。

## 受け入れ基準

- `experiments/exp034_public_sel15_pf_meta_stack/` に config、settings、train notebook、audit script がある。
- audit script は original-fold OOF と deterministic well-hash holdout の両方を実行できる。
- exp026-style base prediction は各 holdout split で validation wells を training files から除外して作る。
- controls として `exp026_pseudo_tail_bucket_shrink`、`public_pf_selector`、`pf090_hold010` を記録する。
- fixed blend、ridge residual meta、shallow HGB residual meta の候補を比較できる。
- overall、distance bucket、split、well metrics と source summary を artifacts に保存する。
- `validate_experiment.py` と Kaggle train notebook preparation が通る。
