# 要件

## 依頼

`KAGGLE_DIRECTION.md` のアイデアバックログ先頭にある `exp010_trajectory_drift_ablation` を実装する。

## 制約

- Kaggle Notebook 実行を正とし、ローカル notebook 実行はしない。
- hidden test に存在しない train-only formation columns は使わない。
- exp008 で悪化した GR NCC と exp009 で悪化した formation guide は disabled のままにする。
- 同一 well を train/valid に分割しない。
- `TVT_input` の evaluation zone の真値は training target / CV metric 以外に使わない。

## 受け入れ基準

- `experiments/exp010_trajectory_drift_ablation/` が作成され、train/inference notebook 名が exp010 に揃っている。
- trajectory direction / slope / full の ablation variants が `config.yaml` に定義されている。
- `baseline.py` に inference-safe trajectory feature columns と feature sets が実装されている。
- `scripts/validate_experiment.py`、`py_compile`、`ruff check`、`pytest` が通る。
- Kaggle 用 train / inference notebook を `--strict` で prepare できる。
