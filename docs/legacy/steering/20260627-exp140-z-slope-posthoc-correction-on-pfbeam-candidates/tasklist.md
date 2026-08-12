# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering docs を作成した。
- `exp140_z_slope_posthoc_correction_on_pfbeam_candidates` を exp099 から派生作成した。
- `config.yaml` を Z-slope posthoc audit 用に更新した。
- `z_slope_posthoc_correction_on_pfbeam_candidates.py` を実装した。
- train / inference notebook を exp140 用に更新した。
- 再現性設計を `design.md` に記入した。
- `py_compile`、`ruff check`、notebook JSON check を通した。
- `make validate-exp EXP=exp140_z_slope_posthoc_correction_on_pfbeam_candidates` を通した。
- Kaggle train package を `--strict` で作成した。
- Kaggle train v1 を push し、notebook metadata 不足による papermill failure を確認した。
- train / inference notebook に `kernelspec.name=python3` と `language_info` metadata を追加した。
- Kaggle train v2 を push し、`KernelWorkerStatus.COMPLETE` を確認した。
- output を `kaggle/output/train_v2` に取得し、metrics / SHA / 解釈を記録した。
- best Z-slope variant が `likpf_mean` から +0.002252946 RMSE 悪化したため、inference port / submit 不採用と判断した。
