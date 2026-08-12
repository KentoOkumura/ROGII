# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- exp184 heatmap compact add-only steering を作成した。
- exp188 add-only scaffold から実験ディレクトリを作成した。
- `config.yaml` を exp148 parent、exp184 selector parent、exp182 heatmap source に更新した。
- `heatmap_selector_compact_addonly_on_exp148.py` に exp184 selected path loader、exp182 heatmap compact feature generator、禁止列ガードを実装した。
- train / inference percent source を exp184 heatmap compact 用に更新した。
- README、result、metrics placeholder、SESSION_NOTES を初期状態に更新した。
- Jupytext で train/inference `.ipynb` を再生成した。
- `py_compile`、`ruff --select F821`、Jupytext `--test`、`make validate-exp` は pass。
- exp072 先頭 100 行の local helper smoke で exp184 selected path、exp182 heatmap compact、exp148 OOF 差分込み 31 features の生成を確認した。
- 検証結果を `SESSION_NOTES.md` に追記した。
- ユーザー指示により Kaggle train runtime を CPU に変更する方針へ更新した。
- `train_lgb0` / `train_lgb1` / `train_lgb2` の prepared metadata が `enable_gpu=false` であることを確認した。
- `train_lgb0` / `train_lgb1` / `train_lgb2` を Kaggle に push し、version 1 が RUNNING になったことを確認した。
- Kaggle split train 3本が version 1 で COMPLETE になったことを確認した。
- split 3本の output を取得し、chunked streaming で横断 `lgb_mean` ensemble CV を計算した。
- cross-split `lgb_mean` RMSE 8.604130846 で exp148 anchor より悪化したため、`completed_train_side_rejected_no_submit` として閉じた。
