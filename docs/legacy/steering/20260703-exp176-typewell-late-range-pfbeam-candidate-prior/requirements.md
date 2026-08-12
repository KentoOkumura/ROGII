# 要件

## 依頼

`KAGGLE_DIRECTION.md` のバックログ `typewell_late_range_pfbeam_candidate_prior` を実装する。

exp174 で ML 予測の late-range hard clip / shrink は no-op または悪化だったため、typewell range prior は hard invalid / direct clip ではなく、PF/Beam candidate selector / verifier の弱い特徴量として扱う。

## 制約

- Route: `ensemble`
- PF/Beam 候補の再生成はしない。
- PF/Beam 候補を hard drop、hard invalid、direct clip しない。
- exp157 と同じ candidate set、GroupKFold、LightGBM ranker 構成を維持し、差分は `candidate_pct` / `known_last_pct` 系特徴量に限定する。
- `candidate_pct` threshold は事前固定し、validation true TVT、oracle best、true-error rank で選ばない。
- 再現性: `docs/06_reproducibility.md` に従い、Kaggle bootstrap、input cache SHA、model manifest、prediction SHA を記録できる構成にする。

## 受け入れ基準

- `docs/legacy/steering/20260703-exp176-typewell-late-range-pfbeam-candidate-prior/` に要件、設計、タスクがある。
- `experiments/exp176_typewell_late_range_pfbeam_candidate_prior/` に config、実装 module、train / inference notebook、記録ファイルがある。
- `config.yaml` に `experiment.route: ensemble` と `ranker.typewell_late_range_prior` がある。
- train notebook は設定確認、input check、typewell prior setup、ranker audit、metrics 表示をセル単位で追える。
- 静的検証、Jupytext 変換検証、synthetic smoke、`validate_experiment.py` が通る。
- Kaggle train push 前に、active variant 数、LightGBM family 数、fold 数、合計 booster 数、control 再学習の有無を `SESSION_NOTES.md` に記録する。
