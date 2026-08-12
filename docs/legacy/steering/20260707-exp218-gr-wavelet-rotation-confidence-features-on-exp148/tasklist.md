# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- 再現性設計を `design.md` に記入した。
- `exp190` から `exp218` 実験ディレクトリを作成した。
- config / notebook / helper module の実験名と feature group を `gr_wavelet_rotation_confidence` へ差し替えた。
- DWT approx/detail/residual、FFT rotation energy、raw-vs-denoised NCC/cost gap、candidate disagreement interaction を生成する実装を追加した。
- train-side GPU cost plan を `SESSION_NOTES.md` に記録した。
- GRWR feature builder smoke、Jupytext 変換、`py_compile`、`ruff`、`validate_experiment.py` を実行した。
- Kaggle push 前に metadata と bootstrap 内 config の整合を確認した。
- Kaggle train v1 を push し、initial status `RUNNING` を確認した。
- Kaggle train v1 の `COMPLETE` を確認した。
- Kaggle output artifact を取得し、CV / bucket / by-well / feature importance を確認した。
- prediction SHA、feature schema SHA、model manifest SHA を `metrics.json` / `SESSION_NOTES.md` に記録した。
- exp148 OOF viewer との streaming id join で exp218 差分を確認した。
- current-test GRWR feature generation と saved-booster inference を実装した。
- Kaggle inference v1 を push し、`COMPLETE` と `submission.csv` 生成を確認した。
- `submission.csv` を sample submission と照合し、submit-check PASS を確認した。
- Submission ref `54457577` の scoring 完了を確認し、Public LB 7.843 を記録した。
- exp218 を ML route submitted anchor として記録した。
