# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- なし。

## 完了

- steering requirements / design を作成。
- exp234 から exact HMM、comparison、fold-safe residual-scale helper を継承。
- scalar-first stage、variance shrinkage、finite alpha、single-variant guard を実装。
- inference no-output contract を実装。
- Python compile、F821、Jupytext convert / test を通過。
- strict experiment validation と canonical train/inference package prepare を通過。
- `KAGGLE_DIRECTION.md` の未着手 backlog から実装済み・train待ちへ移動。
- Kaggle scalar-control v1のNumba thread初期化失敗を診断・修正。
- 同じcanonical kernelのv2を完走し、overall、distance、hidden-like、by-well、step-deltaを記録。
- scalar HMM RMSE `8.361307776`（exp218比`-0.114496982`、exp234比`-0.065923625`）を確認。
- alpha `0.25` v3をscale fit 5 / HMM 1 / booster 0で完走し、fold overlap 0とscale guard passを確認。
- alpha 0.25 RMSE `8.351122273`（scalar比`-0.010185503`）とmixed secondary guardを記録。
- alpha 0.50 v4を完走し、RMSE `8.336863897`（alpha 0.25比`-0.014258376`）を確認。
- 有限ablationを終了し、追加grid / inference / submissionなしと判定。
- ユーザー判断で方向性をclosedとし、全stageをdisabledに変更。
