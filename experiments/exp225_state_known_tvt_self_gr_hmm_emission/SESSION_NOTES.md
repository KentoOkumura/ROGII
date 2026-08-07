# exp225_state_known_tvt_self_gr_hmm_emission セッションノート

## 目的

`state_known_tvt_self_gr_hmm_emission` backlog を実装する。`exp209` exact HMM の typewell GR emission を主軸にし、同一 horizontal well の known prefix から作る `TVT_input -> GR` 曲線を、HMM candidate state が known-prefix TVT 範囲内にある場合だけ弱い clipped boost として足す train-side diagnostic を作る。

## 現在の状態

- Route: `ensemble`
- 状態: Kaggle train v1 完了 / 不採用
- CV: 14.212954500008003
- LB: 未提出
- GPU cost: なし。CPU-only HMM feature generation audit。
- Active variants: 1
- LightGBM config count: 0
- Fold count: 0
- Booster count: 0
- Parent/control retraining: なし
- Inference / submit: なし

## 実装メモ

- 親は `exp223_joint_typewell_self_gr_hmm_likelihood_probe`。HMM base は `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`。
- exp072 full replay cache は再生成せず、保存済み exp072 cache を比較基準として読む。
- `exact_hmm_smoother.py` に `state_known_tvt_curve` self-GR surface を追加。
- self-GR curve は finite `TVT_input` と finite `GR` を持つ known prefix row だけから作る。
- HMM candidate state `grid[j]` が known-prefix TVT 範囲外なら self-GR boost は 0、つまり neutral。
- active variant は `alpha=[0.07]` x `clip=[1.0]` x `mode=[boost_only]` = 1。
- model/config/fold/booster count は 0。

## コマンドログ

```bash
.venv/bin/python scripts/new_steering.py --experiment exp225_state_known_tvt_self_gr_hmm_emission
.venv/bin/python scripts/new_experiment.py --name exp225_state_known_tvt_self_gr_hmm_emission --source experiments/exp223_joint_typewell_self_gr_hmm_likelihood_probe
```

- result: scaffold 作成済み。
- note: 初回は steering 作成と experiment 作成を並列に実行したため steering check が先行して一度失敗。steering 作成後に new_experiment を再実行して成功。

## 次の確認

```bash
.venv/bin/python -m py_compile \
  experiments/exp225_state_known_tvt_self_gr_hmm_emission/settings.py \
  experiments/exp225_state_known_tvt_self_gr_hmm_emission/exact_hmm_smoother.py \
  experiments/exp225_state_known_tvt_self_gr_hmm_emission/feature_cache.py \
  experiments/exp225_state_known_tvt_self_gr_hmm_emission/direct_hmm_comparison.py \
  experiments/exp225_state_known_tvt_self_gr_hmm_emission/joint_cache_generation.py \
  experiments/exp225_state_known_tvt_self_gr_hmm_emission/exp072_feature_cache.py \
  experiments/exp225_state_known_tvt_self_gr_hmm_emission/exp225_state_known_tvt_self_gr_hmm_emission_train.py \
  experiments/exp225_state_known_tvt_self_gr_hmm_emission/exp225_state_known_tvt_self_gr_hmm_emission_inference.py
.venv/bin/ruff check experiments/exp225_state_known_tvt_self_gr_hmm_emission --select F821
python3 -m json.tool experiments/exp225_state_known_tvt_self_gr_hmm_emission/metrics.json
python3 scripts/validate_experiment.py --experiment exp225_state_known_tvt_self_gr_hmm_emission
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 experiments/exp225_state_known_tvt_self_gr_hmm_emission/exp225_state_known_tvt_self_gr_hmm_emission_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 experiments/exp225_state_known_tvt_self_gr_hmm_emission/exp225_state_known_tvt_self_gr_hmm_emission_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp225_state_known_tvt_self_gr_hmm_emission/exp225_state_known_tvt_self_gr_hmm_emission_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp225_state_known_tvt_self_gr_hmm_emission/exp225_state_known_tvt_self_gr_hmm_emission_inference.py
```

- result: PASS。
- variant_count: 1
- variant_names: `hmm_selfgr_state_known_tvt_curve_boost_only_a070_c100`
- feature_count: 13
- expected_feature_count: 13
- `scripts/validate_experiment.py --experiment exp225_state_known_tvt_self_gr_hmm_emission`: PASS
- `ruff --select F821`: PASS
- Jupytext train/inference conversion and `--test`: PASS
- Synthetic state mask unit check: PASS。known TVT 範囲外の grid columns は self-GR centered log-likelihood が 0、範囲内 columns は非ゼロ。
- Short-prefix smoothing guard: `curve_smooth_window` を curve 長で cap し、短い prefix で全体中央値に潰れないよう修正。

## Kaggle train push 前の実行内容

- active variants: 1 (`hmm_selfgr_state_known_tvt_curve_boost_only_a070_c100`)
- LightGBM config count: 0
- fold count: 0
- total boosters: 0
- parent/control retraining: なし
- GPU: なし
- inference / submit: なし

## Kaggle train v1 run

User request: Kaggle で実行する。

- kernel id: `kentookumura/exp225-state-known-tvt-self-gr-hmm-emission-train`
- title: `exp225 state known tvt self gr hmm emission train`
- URL: https://www.kaggle.com/code/kentookumura/exp225-state-known-tvt-self-gr-hmm-emission-train
- version: 1
- id_no: 126463088
- active variants: 1 (`hmm_selfgr_state_known_tvt_curve_boost_only_a070_c100`)
- runtime: CPU, GPU off, internet off
- kernel sources: `kentookumura/exp072-exp063-full-replay-feature-cache-train`, `kentookumura/exp115-hidden-like-spatial-holdout-from-ppt-train`
- model/config/fold/booster count: 0 / 0 / 0 / 0
- parent/control retraining: なし
- inference / submit: なし

```bash
python3 scripts/prepare_kaggle_notebooks.py --experiment exp225_state_known_tvt_self_gr_hmm_emission --notebook train --kernel-id kentookumura/exp225-state-known-tvt-self-gr-hmm-emission-train --title "exp225 state known tvt self gr hmm emission train" --run-on-push --strict
kaggle kernels push -p experiments/exp225_state_known_tvt_self_gr_hmm_emission/kaggle/train
kaggle kernels pull kentookumura/exp225-state-known-tvt-self-gr-hmm-emission-train -p /tmp/kaggle-pull/exp225-state-known-tvt-self-gr-hmm-emission-train-v1 -m
kaggle kernels logs kentookumura/exp225-state-known-tvt-self-gr-hmm-emission-train
kaggle kernels status kentookumura/exp225-state-known-tvt-self-gr-hmm-emission-train
```

- result: push PASS。`Kernel version 1 successfully pushed`。
- metadata pull: PASS。`id` / `title` slug は一致。`enable_gpu=false`、`enable_internet=false`、`competition_sources=["rogii-wellbore-geology-prediction"]`。
- CLI logs: 実行中は空。この環境では running 中 logs が空のことがあるため、失敗扱いにしない。
- initial status after push: `KernelWorkerStatus.RUNNING`。最終状態は下の result section で `COMPLETE` として確認済み。

## Kaggle train v1 result

User reported completion; CLI status confirmed `KernelWorkerStatus.COMPLETE`。詳細 metrics と SHA を確認するため output archive を `/tmp/kaggle-output/exp225-state-known-tvt-self-gr-hmm-emission-train-v1` に取得した。

```bash
kaggle kernels status kentookumura/exp225-state-known-tvt-self-gr-hmm-emission-train
kaggle kernels logs kentookumura/exp225-state-known-tvt-self-gr-hmm-emission-train
kaggle kernels output kentookumura/exp225-state-known-tvt-self-gr-hmm-emission-train -p /tmp/kaggle-output/exp225-state-known-tvt-self-gr-hmm-emission-train-v1
```

- result: COMPLETE。
- rows / wells: 3,783,989 / 773
- elapsed: 17,310.949 sec (約 4h48m31s)
- best overall: exp072 `likpf_mean` RMSE 11.594897668、MAE 7.067632583、within10 0.772802194
- state-known self-GR HMM: `hmm_selfgr_state_known_tvt_curve_boost_only_a070_c100`
- RMSE: 14.212954500
- MAE: 8.290740861
- bias: -5.651001931
- within10: 0.726438423
- delta vs exp072 `likpf_mean`: RMSE +2.618056832、MAE +1.223108278、within10 -0.046363771

Distance bucket delta RMSE vs exp072 `likpf_mean`:

- `000_050`: -0.235925
- `050_100`: -0.316982
- `100_250`: +0.030487
- `250_500`: +0.311389
- `500_1000`: +0.883882
- `1000_plus`: +2.931795

Hidden-like delta RMSE vs exp072 `likpf_mean`:

- `verification_like_spatial`: +2.937794
- `verification_like_typewell_purged`: +2.842109

By-well delta vs exp072 `likpf_mean`:

- improved / worsened: 379 / 394
- mean delta: +1.275998
- median delta: +0.032908
- p90 / p95 delta: +11.517473 / +17.949564
- max regression: `2fd68f7b` +49.423573 RMSE
- next regressions: `8d5d46d7` +49.278663、`b19b0395` +48.319029
- max improvement: `7987f2f2` -42.177293 RMSE

Diagnostics:

- step delta: mean 0.009788、p95 0.035、p99 0.061、5.0/10.0/25.0 超過率 0.0。
- self-GR state-valid rate: mean 0.529194、median 0.513089、min 0.497382、max 0.756356。
- state-valid rate 上位 bucket は delta RMSE -0.568338 で改善するが、全体と hidden-like の悪化を相殺しない。
- feature_count: 13
- active variants: 1
- model/config/fold/booster count: 0 / 0 / 0 / 0
- parent/control retraining: なし
- inference / submit: なし

SHA:

- train_features_decompressed: `c7ab88ad8678d1b70268fbfa91b69622b04225e7cb9aa4075609ff34b879640a`
- train_features_gzip: `02223436c8a5a77011f4e20f88bd823dfcc29b6c3e4dc5336fac947af8806308`
- joint_summary: `32fb1d30d439da2bc2792bcd4e2f116904f8c712957b560c10d3a60c4f58cab9`
- feature_schema: `4916cf4f31850251604e617bca34e303b624bbdc954c1b1835438da2e07dcb23`
- by_well_generation_summary: `522d48f8ef9d819fcd6f6d780f6cd092c5cc99f53f181cd0168b74d97ed66713`
- overall_metrics: `9d69704eb75879309977d55f570481943bf50e0900b08d0a8cc90f46fc37685e`
- distance_bucket_metrics: `82b0d1d6a47defb8a8e27cb43092a54942b345ea45a469a1ebf8816d1d36111e`
- hidden_like_metrics: `93a74f4671673be6e888550df2f15e335a44b2afc4e40192fe277d0d4fc7feb8`
- by_well_delta: `f5f12aa26fc3d15c5e4a3851c52fbfa3c2ee6680fa080783a9370a1754ec5d2d`

判定:

state-known trigger の実装は成立したが、RMSE、hidden-like、longtail、worst-well がすべて不採用水準。追加 alpha / sigma grid、raw-test regeneration、inference、submit は行わない。
