# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- `docs/legacy/steering/20260708-exp224-well-scaled-z-dz-features-on-exp218/` を作成した。
- `exp218_gr_wavelet_rotation_confidence_features_on_exp148` から `exp224_well_scaled_z_dz_features_on_exp218` を作成した。
- `build_well_scaled_z_dz_features()` を実装した。
- train / inference の feature assembly に `well_scaled_z_dz` feature group を接続した。
- `config.yaml` に route、lineage、feature config、active variant、planned booster 数を記録した。
- `README.md` / `SESSION_NOTES.md` / `result.md` / `metrics.json` を exp224 初期状態に更新した。
- Jupytext で train / inference `.ipynb` を再生成した。
- py_compile / ruff F821,F401 / validate_experiment を通した。
- synthetic feature builder smoke を通した。
- `prepare_kaggle_notebooks.py --strict` で train / inference package を作成した。
- CPU split 要件に合わせて `train_lgb0` / `train_lgb1` / `train_lgb2` 実装へ変更した。
- `prepare_kaggle_notebooks.py --strict` で CPU `train_lgb0` / `train_lgb1` / `train_lgb2` package を作成し、metadata `enable_gpu=false` を確認した。
- Kaggle CPU split train push 前に active variant/config/fold/booster 数を再確認した。
- `train_lgb0` / `train_lgb1` / `train_lgb2` を Kaggle で実行し、3 本とも `KernelWorkerStatus.COMPLETE` を確認した。
- split OOF を取得し、3-config `lgb_mean` を集計した。RMSE TVT は 8.538687042。
- CV 悪化により不採用とし、`SESSION_NOTES.md` / `result.md` / `metrics.json` に記録した。
