# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- `.steering/20260705-exp198-exact-replacement-prune-on-exp148/` を作成した。
- 再現性設計を `design.md` に記入した。
- exp148 から `exp198_exact_replacement_prune_on_exp148` を作成した。
- config に route、lineage、drop list、active variant、GPU コストガードを記録した。
- train notebook / Jupytext source を exp198 名に更新し、active feature list から 17 列だけを除外する実装を入れた。
- inference notebook は初回 train 評価待ちとして exp198 名・親情報に整えた。
- Jupytext 変換、構文チェック、`ruff --select F821`、`validate-exp` を通した。
- Kaggle train package を strict prepare し、metadata と bootstrap 内 config の整合を確認した。
- Kaggle push 前に active variant 1、LightGBM config 3、fold 5、合計 booster 15、control 再学習なしを `SESSION_NOTES.md` に記録した。
- Kaggle train v1 を完了し、CV、feature count、bucket、by-well、生成物 SHA を記録した。
- train-side supported と判断した。inference / submission は未実行のため、submission SHA は対象外として扱う。
