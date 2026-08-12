# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- 要件と再現性設計を記入した。
- `experiments/exp089_pf_beam_disagreement_sample_weight/` を作成した。
- `config.yaml` を PF/Beam confidence feature / sample weight ablation 用に更新した。
- `pf_beam_disagreement_sample_weight.py` に confidence feature 生成、rank-based sample weight、LightGBM CV、artifact 保存を実装した。
- train notebook を exp089 用に更新した。
- `py_compile`、notebook JSON validation、ruff、`validate_experiment.py` を通した。
- synthetic frame で confidence feature / sample weight smoke test を通した。
- Kaggle train package を作成し、metadata と bootstrap manifest を確認した。
- Kaggle train v1 を push した。
- Kaggle train v1 の完了を確認した。
- output を取得し、feature source SHA、OOF prediction SHA、model count、bucket / by-well 悪化を記録した。
- well-level guard と mid-distance bucket 悪化により submit candidate にはしないと判断した。
