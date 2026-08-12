# 要件

## 依頼

`KAGGLE_DIRECTION.md` のアイデアバックログ先頭にある `exp009_formation_surface_guide` を実装する。
train-only formation columns (`ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`) を hidden test の直接特徴には使わず、fold-safe な structural guide に変換して residual model の補助特徴として検証する。

## 制約

- Kaggle Notebook 実行を正とし、ローカル notebook 実行はしない。
- 同一 well は train/valid fold をまたいで混入させない。
- valid fold の formation columns は guide 学習にも特徴生成にも使わない。
- hidden test に存在しない formation columns を inference の入力として要求しない。
- GR NCC は exp008 で悪化したため、この実験では再投入しない。

## 受け入れ基準

- `exp009_formation_surface_guide` の実験フォルダと train/inference notebook がある。
- KNN surface guide が fold ごとに train fold だけから fit され、valid/test には推定 surface 由来の特徴だけが渡る。
- ablation に `control_exp002_all`、`control_exp003_no_gr`、formation guide 追加 variants が含まれる。
- `validate_experiment.py`、`ruff check`、`py_compile`、`pytest` が通る。
- `SESSION_NOTES.md` に実装内容、検証コマンド、次の Kaggle 実行手順が記録される。
