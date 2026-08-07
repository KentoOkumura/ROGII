# exp153_full_rank_slot_addonly_on_exp092 セッションノート

## 2026-06-28 実装

- 当初、backlog 名 `exp098_full_rank_slot_addonly_on_exp092` をそのまま実験名として作成してしまったが、既存 `exp098_selector_rank_slot_features_on_exp073` と番号が重複するため、正式実験名を `exp153_full_rank_slot_addonly_on_exp092` に変更した。
- `.steering/20260628-exp153-full-rank-slot-addonly-on-exp092/` を作成。
- `experiments/exp153_full_rank_slot_addonly_on_exp092/` を `exp139_exp092_exp098_small_rank_slot_merge` から作成し、full rank-slot add-only 用に差し替えた。
- 親実験は `exp092_u_projection_correction_disagreement_fullrun`、rank-slot source parent は `exp098_selector_rank_slot_features_on_exp073`、cache は `exp072_exp063_full_replay_feature_cache`。
- exp092 の `projection_correction` / `u_disagreement` は維持し、exp098 と同じ target-free `rank_slot_delta` / `rank_slot_identity_score` / `rank_slot_u_projection` / `rank_slot_u_disagreement` を add-only で追加する。
- Candidate TVT path の direct selector、soft average、blend、postprocess replacement、target 変更は入れない。
- Colab runner `exp153_full_rank_slot_addonly_on_exp092_colab_train.ipynb` を追加。Drive 上の exp072 cache を `/content/rogii_cache/exp072_artifacts/` にコピーしてから background full train を実行し、log / PID / latest summary を Drive の `colab_runs/` に残す。
- Full rank-slot feature groups は既存 generator の `rank_slot_delta` 9、`rank_slot_identity_score` 26、`rank_slot_u_projection` 21、`rank_slot_u_disagreement` 8 の計 64 columns を想定する。

## GPU コストガード

- active variant 数: 1 (`u_projection_full_rank_slot_addonly`)
- LightGBM config 数: 3 (`lgb0`, `lgb1`, `lgb2`)
- fold 数: 5
- 合計 booster 数: 15
- exp092 control 再学習: なし
- baseline は保存済み exp092 `lgb1` CV 9.322479896 / Public LB 8.350 を参照する。

## 検証ログ

- `python3 -m json.tool experiments/exp153_full_rank_slot_addonly_on_exp092/exp153_full_rank_slot_addonly_on_exp092_train.ipynb`: PASS
- `python3 -m json.tool experiments/exp153_full_rank_slot_addonly_on_exp092/exp153_full_rank_slot_addonly_on_exp092_inference.ipynb`: PASS
- `python3 -m json.tool experiments/exp153_full_rank_slot_addonly_on_exp092/exp153_full_rank_slot_addonly_on_exp092_colab_train.ipynb`: PASS
- `python3 -m py_compile experiments/exp153_full_rank_slot_addonly_on_exp092/full_rank_slot_addonly_on_exp092.py experiments/exp153_full_rank_slot_addonly_on_exp092/public_notebook_replay_audit.py experiments/exp153_full_rank_slot_addonly_on_exp092/settings.py`: PASS
- `uv run ruff check experiments/exp153_full_rank_slot_addonly_on_exp092/full_rank_slot_addonly_on_exp092.py experiments/exp153_full_rank_slot_addonly_on_exp092/public_notebook_replay_audit.py experiments/exp153_full_rank_slot_addonly_on_exp092/settings.py`: PASS
- `uv run ruff format --check experiments/exp153_full_rank_slot_addonly_on_exp092/full_rank_slot_addonly_on_exp092.py experiments/exp153_full_rank_slot_addonly_on_exp092/public_notebook_replay_audit.py experiments/exp153_full_rank_slot_addonly_on_exp092/settings.py`: PASS
- `make validate-exp EXP=exp153_full_rank_slot_addonly_on_exp092`: PASS

## 2026-06-28 Colab 試行メモ

- 誤命名時点の `exp098_full_rank_slot_addonly_on_exp092` を Colab L4 / 約53GB RAM セッション `exp098-full-rank` で起動した。
- Drive mount、cache copy、LightGBM GPU smoke は PASS。
- Background full train は `rows=3783989`、`features=304`、`configs=3` まで進み、LightGBM training phase に入ったが、Colab セッションが失われたため完了していない。
- この試行は旧 Drive path `experiments/exp098_full_rank_slot_addonly_on_exp092/` のものなので、正式な `exp153_full_rank_slot_addonly_on_exp092` の結果としては扱わない。
- 再実行する場合は Drive 上の `exp153_full_rank_slot_addonly_on_exp092` に同期してから、Colab session 名も `exp153-full-rank` などに変えて開始する。

## 2026-06-28 Colab 公式 run 開始

- Colab session: `exp153-full-rank` (`NVIDIA L4`, RAM 52.96GB)
- Drive root: `/content/drive/MyDrive/Kaggle/ROGII`
- Drive mount は初回 `ValueError: mount failed` だったが、再試行で `Mounted at /content/drive` を確認。
- Drive 上で `project.yml` と exp072 cache `experiments/exp072_exp063_full_replay_feature_cache/artifacts/exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz` を確認。cache size は `2093372344` bytes。
- ローカルの正式 exp153 実装と steering docs を tar package で Colab に upload し、Drive root に展開した。
- exp072 cache を `/content/rogii_cache/exp072_artifacts/` にコピーした。copy time は 34.83 sec、preview shape は `(3, 199)`。
- LightGBM GPU smoke: `lightgbm_gpu_smoke_ok 20`。
- Official full train run:
  - run id: `run_20260628_122618_l4_highmem_local_cache`
  - parent PID: `5640`
  - train PID: `5643`
  - log: `/content/drive/MyDrive/Kaggle/ROGII/experiments/exp153_full_rank_slot_addonly_on_exp092/colab_runs/run_20260628_122618_l4_highmem_local_cache_exp153_full_rank_slot_addonly_on_exp092_full_train.log`
  - completion marker: `/content/drive/MyDrive/Kaggle/ROGII/experiments/exp153_full_rank_slot_addonly_on_exp092/colab_runs/latest_done_summary.json`
  - failure marker: `/content/drive/MyDrive/Kaggle/ROGII/experiments/exp153_full_rank_slot_addonly_on_exp092/colab_runs/latest_failed.txt`
- 3分時点では log は initial block のみ、`latest_done_summary.json` / `latest_failed.txt` は未作成、train PID `5643` が CPU 使用中。前回も feature/config 行まで時間がかかったため、この時点では中断とは判断しない。
- 約10分時点で正式 exp153 path の log に `{"configs": 3, "features": 304, "rows": 3783989, ...}` が出力された。
- その後 `lgb0` fold 0 が完了。`best_iteration=1219`, `rmse_tvt=8.703523356345105`。
- fold 1 はまだ完了していないが、train PID `5643` は継続中。`nvidia-smi` では NVIDIA L4 上で PID `5643` が GPU memory 1104MiB、GPU util 48% で稼働しているため、学習中と判断する。

## 2026-06-29 Colab 完了確認

- ユーザー指摘どおり、`exp153-cli-run` の CLI session lost 後も Drive 上では full train が完了していた。
- 確認用 session `exp153-check` で Drive mount し、次を確認した。
  - `colab_runs/latest_done_summary.json`: exists, size `9058`
  - `colab_runs/latest_failed.txt`: absent
  - `artifacts/exp153_full_rank_slot_addonly_on_exp092_summary.json`: exists, size `9014`
  - `artifacts/exp153_full_rank_slot_addonly_on_exp092_metrics.csv`: exists, size `6322`
  - `artifacts/exp153_full_rank_slot_addonly_on_exp092_lgb_models/manifest.json`: exists, size `20678`
- Completed run:
  - run id: `run_20260628_133009_l4_highmem_local_cache`
  - status: `train_completed`
  - elapsed_seconds: `10000.986`
  - colab_elapsed_seconds_outer: `10001.721`
  - rows: `3783989`
  - features: `304`
  - active mode: `gpu_repro_guard_dp_threads8`
  - active variant: `u_projection_full_rank_slot_addonly`
- Pooled CV:
  - `lgb0`: RMSE TVT `9.630078943598752`
  - `lgb1`: RMSE TVT `9.388317392953782`
  - `lgb2`: RMSE TVT `9.413605687155531`
  - `lgb_mean`: RMSE TVT `9.423385453890534`
- Best summary row is `lgb_mean` RMSE TVT `9.423385453890534`, prediction SHA `083f4b185718a9b8f08f4b06b030e8568d52f2e1a8bccd94ca516ccfbad0ad31`.
- `exp092` best `lgb1` CV `9.322479895503927` / `lgb_mean` CV `9.343064065995073` より悪化。full rank-slot add-only は不採用。
- 前回 assistant が `colab sessions` の active session 消失だけで「失敗」と判断したのは誤り。Colab CLI session loss は run completion と同義ではなく、Drive-backed done marker を確認してから完了/失敗を判断する。

## 2026-06-29 最小 output 取得

- Kaggle output 取得と同等のローカル監査用として、Drive 上の Colab 生成物から必要最低限の output を取得した。
- 保存先: `experiments/exp153_full_rank_slot_addonly_on_exp092/kaggle/output/colab_run_20260628_133009_l4_highmem_local_cache/`
- 取得したもの:
  - `metrics.json`, `config.yaml`, `result.md`, `SESSION_NOTES.md`
  - `colab_runs/latest_run.json`
  - `colab_runs/latest_done_summary.json`
  - `colab_runs/run_20260628_133009_l4_highmem_local_cache_exp153_full_rank_slot_addonly_on_exp092_full_train.log`
  - `artifacts/exp153_full_rank_slot_addonly_on_exp092_summary.json`
  - `artifacts/exp153_full_rank_slot_addonly_on_exp092_metrics.csv`
  - `artifacts/exp153_full_rank_slot_addonly_on_exp092_by_well.csv`
  - `artifacts/exp153_full_rank_slot_addonly_on_exp092_bucket_metrics.csv`
  - `artifacts/exp153_full_rank_slot_addonly_on_exp092_projection_feature_summary.csv`
  - `artifacts/exp153_full_rank_slot_addonly_on_exp092_rank_slot_feature_summary.csv`
  - `artifacts/exp153_full_rank_slot_addonly_on_exp092_feature_importance_mean.csv`
  - `artifacts/exp153_full_rank_slot_addonly_on_exp092_feature_importance_mean_top.png`
  - `artifacts/exp153_full_rank_slot_addonly_on_exp092_feature_schema.csv`
  - `artifacts/exp153_full_rank_slot_addonly_on_exp092_lgb_models_manifest.json`
  - `minimal_output_manifest.json`
- 意図的に取得しなかったもの:
  - `predictions.csv.gz`: 予測全量で重いため、今回の完了確認・不採用判断には不要。
  - LightGBM model `.txt`: submit / inference port をしないため、ローカル監査には manifest だけで足りる。
- `python3 -m json.tool` で `metrics.json` と `colab_runs/latest_done_summary.json` の JSON 妥当性を確認済み。
