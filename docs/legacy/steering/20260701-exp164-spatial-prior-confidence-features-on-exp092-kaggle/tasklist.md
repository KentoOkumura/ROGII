# タスクリスト

## TODO

- Kaggle train 完了後、logs / notebook cell output から pooled RMSE、fold metrics、generated artifacts path を `SESSION_NOTES.md` と `result.md` に記録する。
- train result が改善した場合だけ、raw-test spatial prior parity と inference 実装を同じ exp164 内で検討する。
- output 取得が必要になった場合だけ、feature content SHA、prediction SHA、model manifest SHA をローカルで確認する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering docs を作成した。
- `exp164_spatial_prior_confidence_features_on_exp092_kaggle` を作成した。
- `exp159` の Colab runner / manual upload / checkpoint 前提を採用しない方針に切り替えた。
- active variant / LightGBM config / fold / booster 数を `SESSION_NOTES.md` に記録した。
- Jupytext percent 形式の train / inference notebook source を追加し、`.ipynb` を再生成した。
- Kaggle train package を `run_on_push=true` で生成した。
- CPU 実行用に `train_lgb0` / `train_lgb1` / `train_lgb2` notebook へ分割した。
- 3 本の CPU notebook を Kaggle に push し、version 2 を RUNNING にした。
