# exp230_dtw_typewell_warp_hmm_emission_correction セッションノート

## 目的

`dtw_typewell_warp_hmm_emission_correction` backlog を、ユーザー指定どおり `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation` を親として実装する。full horizontal GR と typewell GR の constrained DTW warp は、直接予測候補ではなく exp209 exact HMM の補助 emission としてだけ使う。

## 現在の状態

- 状態: Kaggle train version 2 完了・不採用
- Route: `pf_beam`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- active variants: `hmm_dtw_a005_s1200`, `hmm_dtw_a010_s1200`
- LightGBM config / fold / booster: 0 / 0 / 0
- parent/control retraining: なし
- GPU: false
- inference / submit: なし

## 実装メモ

- `scripts/new_steering.py --experiment exp230_dtw_typewell_warp_hmm_emission_correction`
- `scripts/new_experiment.py --name exp230_dtw_typewell_warp_hmm_emission_correction --source experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- `exact_hmm_smoother.py`
  - exp209 HMM を維持。
  - constrained DTW path builder を追加。
  - stable SHA seed の jitter で DTW anchor stability を診断。
  - `emission_ll += alpha * dtw_ll` のみ追加。
- `feature_cache.py`
  - `model.dtw_emission` を HMM generator に渡す。
- `direct_hmm_comparison.py`
  - 複数 `*_mean_tvt` HMM candidates と exp115 hidden-like metrics に対応。
- `config.yaml`
  - exp072 full replay regeneration を無効化。
  - saved exp072 cache を baseline として使用。

## 実行コストガード

- DTW-HMM variants: 2
- LightGBM configs: 0
- folds: 0
- total boosters: 0
- Kaggle GPU: false
- exp209 / exp072 control retraining: なし

## 検証ログ

- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 experiments/exp230_dtw_typewell_warp_hmm_emission_correction/exp230_dtw_typewell_warp_hmm_emission_correction_train.py`: pass
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 experiments/exp230_dtw_typewell_warp_hmm_emission_correction/exp230_dtw_typewell_warp_hmm_emission_correction_inference.py`: pass
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp230_dtw_typewell_warp_hmm_emission_correction/exp230_dtw_typewell_warp_hmm_emission_correction_train.py`: pass
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp230_dtw_typewell_warp_hmm_emission_correction/exp230_dtw_typewell_warp_hmm_emission_correction_inference.py`: pass
- `.venv/bin/python -m py_compile ...`: pass
- `.venv/bin/ruff check experiments/exp230_dtw_typewell_warp_hmm_emission_correction --select F821`: pass
- `.venv/bin/python scripts/validate_experiment.py --experiment exp230_dtw_typewell_warp_hmm_emission_correction`: pass
- `.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp230_dtw_typewell_warp_hmm_emission_correction --notebook train --kernel-id kentookumura/exp230-dtw-typewell-warp-hmm-emission-correction-train --title "exp230 dtw typewell warp hmm emission correction train" --run-on-push --strict`: pass

## Kaggle package

- path: `experiments/exp230_dtw_typewell_warp_hmm_emission_correction/kaggle/train`
- initial long kernel id: `kentookumura/exp230-dtw-typewell-warp-hmm-emission-correction-train`
- pushed kernel id: `kentookumura/exp230-dtw-hmm-emission-train`
- title: `exp230 dtw hmm emission train`
- latest pushed version: 2
- status after v2 push: `KernelWorkerStatus.RUNNING`
- pre-completion checked status: `KernelWorkerStatus.RUNNING` at 2026-07-10 06:50 JST
- run_on_push: true
- enable_gpu: false
- enable_internet: false
- kernel_sources:
  - `kentookumura/exp072-exp063-full-replay-feature-cache-train`
  - `kentookumura/exp115-hidden-like-spatial-holdout-from-ppt-train`
- note: initial long slug was rejected by Kaggle API 400, so the package was regenerated with the shorter slug above.
- note: `kaggle kernels logs` and a 5 minute `logs -f` follow returned no notebook logs yet while the worker remained running.
- note: after v1 push, local source/package was updated so notebook-written `metrics.json` uses `kaggle_train_completed` after successful completion. This one-line status writer cleanup was not pushed as a separate v2.
- note: on 2026-07-10 06:44 JST, v1 still appeared as `RUNNING`, `kaggle kernels files` returned no output files, and logs were empty. The latest local package was pushed to the same kernel id as version 2 with `run_on_push=true`.
- note: after v2 push, a 5 minute `kaggle kernels logs -f --interval 20` follow returned no logs. At 2026-07-10 06:50 JST, `status` was still `RUNNING`, `logs` was empty, and `files` returned no output files.
- note: `kaggle kernels list --mine --search exp230-dtw-hmm-emission-train` showed `lastRunTime` = 2026-07-09 21:44:05 UTC, matching the v2 push time.

## Kaggle v2 結果

- checked at: 2026-07-11 00:24 JST
- `kaggle kernels status kentookumura/exp230-dtw-hmm-emission-train`: `KernelWorkerStatus.COMPLETE`
- `kaggle kernels logs kentookumura/exp230-dtw-hmm-emission-train > /tmp/exp230_kaggle_logs.json`: pass
- `kaggle kernels output kentookumura/exp230-dtw-hmm-emission-train -p /tmp/exp230_output_metrics --file-pattern 'exp230_dtw_hmm_vs_exp072_.*\.(csv|json)|exp230_dtw_hmm_generation_summary\.json|metrics\.json' -o`: pass
- output archive policy: train-side readout は logs を主根拠にし、hidden-like / bucket / by-well CSV の確認が必要だったため metrics 系ファイルだけ取得した。large train feature cache は取得していない。
- rows / wells: 3,783,989 / 773
- elapsed: 36,768.8 sec
- best overall: `exp072_likpf_mean` RMSE 11.594897668 / MAE 7.067632583 / within10 0.772802194
- best DTW-HMM: `hmm_dtw_a005_s1200` RMSE 13.611292323 / MAE 7.696311594 / within10 0.753860278
- delta vs exp072 `likpf_mean`: RMSE +2.016394654 / MAE +0.628679011 / within10 -0.018941916
- `hmm_dtw_a010_s1200`: RMSE 16.435494713, delta +4.840597045
- hidden-like delta for `hmm_dtw_a005_s1200`: verification_like_spatial +1.645723 RMSE, verification_like_typewell_purged +1.740356 RMSE
- distance bucket delta for `hmm_dtw_a005_s1200`: `000_050` -0.279896, `050_100` -0.424749, `100_250` -0.224815, `250_500` -0.082857, `500_1000` +0.128648, `1000_plus` +2.300709
- by-well `hmm_dtw_a005_s1200`: 409 improved / 364 worsened, max regression `b19b0395` +47.803293 RMSE
- step-delta spikes for `hmm_dtw_a005_s1200`: >5 / >10 / >25 ft are all 0
- HMM feature decompressed SHA256: `a939913c786cf07e21feff01eadaf4258eb772384f82e9682921a57abd2893e4`
- decision: completed / rejected. No raw-test regeneration, no inference, no submit, no additional alpha grid.

## 禁止事項

- DTW path TVT の直接採用なし。
- softmax average / hard gate / PF weight 変更なし。
- LB weight search なし。
- 真の tail TVT、oracle best、true-error rank を generation に使わない。

## 次のアクション

1. DTW / elastic registration を使う場合は HMM emission への直接補正ではなく、ML / selector confidence feature または regression guard readout に限定する。
2. exp230 の raw-test regeneration、inference、submit、追加 alpha grid は行わない。
