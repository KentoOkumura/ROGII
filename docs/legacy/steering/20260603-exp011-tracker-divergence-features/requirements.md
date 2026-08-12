# 要件

## 依頼

`KAGGLE_DIRECTION.md` のアイデアバックログ先頭にある `exp011_tracker_divergence_features` を実装する。

## 制約

- Kaggle Notebook 実行を正とし、ローカル notebook 実行はしない。
- stochastic PF は使わず、再現可能な deterministic feature generation に限定する。
- `TVT_input` の既知 prefix、`MD`、`Z`、`GR`、paired typewell `GR` だけから tracker features を作る。
- exp010 で悪化した full trajectory slope/interactions はそのまま再投入しない。
- train-only formation columns は使わない。
- exp002 all-GR と exp003 no-GR control を同じ GroupKFold で再計算する。

## 受け入れ基準

- `experiments/exp011_tracker_divergence_features/` に config、notebook、実装、記録ファイルが揃う。
- `baseline.py` に tracker feature set が追加され、`active_feature_columns` で検証できる。
- `config.yaml` に `control_exp002_all`、`control_exp003_no_gr`、tracker variants が定義される。
- train notebook が `tracker_group_summary.csv` を出力し、hard/no-GR、steep trajectory、high GR missing、long eval group を確認できる。
- `validate_experiment.py`、py_compile、ruff、pytest が通る。
