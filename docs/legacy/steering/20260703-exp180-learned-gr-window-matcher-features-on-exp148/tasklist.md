# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering docs を作成した。
- `exp180_learned_gr_window_matcher_features_on_exp148` を作成した。
- `learned_gr_window_matcher` feature group を実装した。
- train / split train / feature cache notebook source を exp180 用に更新した。
- `scripts/prepare_kaggle_notebooks.py` に `gr_matcher_features` kind を追加した。
- 再現性設計を `design.md` に記入した。
- Jupytext 変換と `--test` を通した。
- `py_compile`、`ruff --select F821`、`make validate-exp` を通した。
- `gr_matcher_features` package を `--strict` で prepare した。
- `kentookumura/exp180-gr-matcher-exp148-features` v1 を push し、RUNNING を確認した。
- `kentookumura/exp180-gr-matcher-exp148-features` v1 の COMPLETE を確認した。
- `train_lgb0` v1 の memory/OOM 系 failure を確認し、memory 対策を入れた v2 を push した。
- `train_lgb1` / `train_lgb2` は canonical slug の `Notebook not found` を回避し、retry slug `train-lgb1-r1` / `train-lgb2-r1` で push した。
- train 3 split の RUNNING を確認した。
- train 3 split の COMPLETE を確認した。
- OOF prediction を取得し、3 split の `lgb_mean` ensemble CV 8.5145263671875 を計算した。
- exp148 `lgb_mean` CV 8.50128118189582 から +0.0132451852916803 悪化したため、inference / submit-check / submit は実行しない判断にした。
- bucket、by-well、feature importance を軽量確認した。hidden-like stress は global OOF negative のため追加実行しない。
