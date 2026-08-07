# タスクリスト

## TODO

- inference output を取得し、`submission.csv` の submit-check を行う。
- 必要なら split OOF predictions を取得し、cross-manifest `lgb_mean` CV / by-well / bucket を確認する。
- 改善した場合のみ inference parity と submit-check を行う。

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering docs を作成した。
- exp162 実験ディレクトリを作成した。
- CPU deterministic mode と `enable_gpu=false` を config に設定した。
- learned likelihood rank-slot feature generator を実装した。
- train / inference の Jupytext percent notebook を作成した。
- CPU timeout mitigation として lgb0 / lgb1 / lgb2 別の train notebook と Kaggle package を作成した。
- inference が 3つの split train manifest を読み込んで `lgb_mean` を平均できるようにした。
- split train_lgb0 / train_lgb1 / train_lgb2 を Kaggle に push し、3本とも完走確認した。
- inference v3 を Kaggle で完走させ、15 booster の `lgb_mean` submission を生成した。
