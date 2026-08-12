# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- `docs/legacy/steering/20260626-exp132-multi-scale-gr-observation-likelihood/` を作成した。
- `experiments/exp132_multi_scale_gr_observation_likelihood/` を exp099 から派生作成した。
- `config.yaml` に route、lineage、leakage policy、multi-scale GR likelihood 設定、expected outputs を記録した。
- `multi_scale_gr_observation_likelihood.py` を実装した。
- train notebook / inference notebook を exp132 用に更新した。
- 再現性設計を `design.md` に記入した。
- `.venv/bin/python -m py_compile` を通した。
- `.venv/bin/ruff check` を通した。
- synthetic cache / raw train による helper smoke を通した。
- `make validate-exp EXP=exp132_multi_scale_gr_observation_likelihood` を通した。
- Kaggle train package を `kentookumura/exp132-msgr-likelihood-train` / title `exp132 msgr likelihood train` で生成した。
- Kaggle train v1 を push した。
- Kaggle output を `experiments/exp132_multi_scale_gr_observation_likelihood/kaggle/output/train_v1` に取得した。
- candidate / rank / bucket / by-well metrics と feature cache SHA を記録した。
- `likpf_mean` を上回る non-oracle scorer / gate がないため、実験を rejected と判定した。
