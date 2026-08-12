# タスクリスト

## TODO

- LB 確認する場合は inference v1 submission を submit する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering docs を作成。
- exp073 実験フォルダを作成。
- exp072 full replay cache を読む LightGBM reproducibility guard script を実装。
- train/inference notebook を exp073 用に更新。
- `py_compile`、`ruff check`、notebook JSON validation、`validate_experiment.py` を通過。
- Kaggle train/inference notebook packages を生成。
- exp072 完了後、exp073 train notebook v1 を Kaggle に push。
- exp073 train v1 の pull existence check を通過。
- exp073 train v1 のログを確認し、train は exp072 cache を読むだけで PF/Beam 再生成をしていないことを確認。
- exp073 train v1 は metrics/model artifact 作成前に手動停止されたため、`stopped_before_metrics` として記録。
- exp072 deterministic v2 完了後、exp073 train v2 を Kaggle に push。
- exp073 train v2 の pull existence check を通過。
- exp073 のスコープを LightGBM train-only ではなく PF/Beam/likelihood-PF test regeneration まで含む end-to-end reproducibility guard に修正。
- exp073 の `public_notebook_replay_audit.py` を exp072 deterministic v2 と同じ stable per-well seed 実装へ差し替え。
- exp073 inference Kaggle package を再生成し、package 内に deterministic PF/Beam seed 実装が含まれることを確認。
- CPU deterministic train package `kaggle/train_cpu` を作成し、metadata と embedded config が CPU 設定であることを確認。
- CPU deterministic train v1 を Kaggle に push し、pull existence check を通過。
- CPU deterministic train v1 の初期ログ / output probe は空で、Kaggle API lag または実行中として記録。
- exp073 GPU train v2 の完了ログと output を取得。
- exp073 GPU train v2 の CV、OOF prediction SHA、model SHA count、runtime、exp072 source SHA を記録。
- exp073 deterministic inference v1 を Kaggle に push。
- exp073 deterministic inference v1 の完了ログと output を取得。
- exp073 deterministic inference v1 の feature SHA、prediction SHA、submission SHA、fallback 0、submit-check PASS を記録。
- exp073 deterministic inference v2 を rerun し、decompressed feature content SHA、prediction SHA、submission SHA、submission CSV bytes が固定されることを確認。
- raw gzip feature file SHA は v1/v2 で異なるため、feature-content determinism key には使わない方針を記録。
- CPU deterministic train v1 の完了ログと output を取得。
- CPU deterministic train v1 の CV、OOF prediction SHA、model SHA count、runtime、exp072 source SHA を記録。
- CPU deterministic train v1 と GPU train v2 の pooled RMSE 差を記録。
- CPU deterministic inference package `kaggle/inference_cpu` を作成し、metadata と embedded config が CPU train source / CPU selected mode / GPU disabled であることを確認。
- CPU deterministic inference v1 を Kaggle に push し、bootstrap ZIP 内 config が古く GPU selected mode のままだったため失敗したことを記録。
- CPU deterministic inference notebook の bootstrap ZIP を CPU package files から再生成。
- CPU deterministic inference v2 を Kaggle に push し、完了 output、SHA、submit-check PASS、GPU-vs-CPU submission diff を記録。
