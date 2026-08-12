# タスクリスト

## TODO

- Kaggle train 実行後、CV、band summary、model SHA、prediction SHA を `SESSION_NOTES.md` / `result.md` に記録する。
- HMM audit 実行後、overall / bucket / hidden-like / by-well / step-delta を記録し、inference port の可否を判断する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering 作成。
- `config.yaml` に route、lineage、quantile train、HMM audit、Kaggle sources、再現性方針を記載。
- quantile LightGBM train helper を実装。
- row-wise sigma 対応の HMM helper を実装。
- HMM audit helper を実装。
- `train` / `train_aggregate` notebook source を実装。
- Jupytext で `train.py` と `train_aggregate.py` を `.ipynb` に変換した。
- `py_compile` と `ruff --select F821` を通した。
- `validate_experiment.py` で exp 構成を確認した。
- Kaggle package を `train` / `train_aggregate` とも生成し、metadata と package-side checks を確認した。
