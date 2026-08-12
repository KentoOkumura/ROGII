# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- 親、route、固定HMM variant、二峰判定、リーク境界、再現性方針を固定した。
- exp221互換 HMM wrapper、posterior shape / segment診断、Jupytext train / disabled inference notebookを実装した。
- static validation、Jupytext round-trip、strict experiment validation、strict Kaggle CPU package生成を完了した。
- Kaggle CPU train-side audit v1 を完了し、HMM input coverage、bimodality rate、decoder
  readout、hidden-like / worst-well guardを記録した。
- posterior mean が RMSE 8.327728486 で最良、二峰 row は 0.9355%、mode mass switch は17回だった。
  MAP / dominant-mode direct decoder、mixture emission、mode-stateの後続は支持されず不採用とした。
