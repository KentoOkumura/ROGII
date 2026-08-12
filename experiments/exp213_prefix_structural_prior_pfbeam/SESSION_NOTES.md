# exp213_prefix_structural_prior_pfbeam セッションノート

## 目的

`prefix_structural_prior_pfbeam` backlog を実装する。`TVT_input + Z` prefix surface を known prefix だけで fit し、PF/Beam の初期状態と遷移 prior に使う train-side audit を作る。

## 現在の状態

- Route: `pf_beam`
- 状態: `completed_train_side_diagnostic_no_submit`
- CV: train-side diagnostic only
- LB: なし
- 注記: train-side scoped audit のみ。LightGBM 学習、inference port、direct replacement、submit は対象外。

## 実行予定

- active variant 数: 4 (`raw`, `structural_slope_only`, `structural_weak`, `structural_base`)
- model/config 数: LightGBM なし、PF/Beam generation audit のみ
- fold 数: 0
- 合計 booster 数: 0
- control / 親実験の再学習: なし
- target wells: 最大 64 wells
- score rows: exp072/099/189/211 と同じ `TVT_input_missing_equivalent_exp063_rows`
- PF config: 240 particles x 8 seeds / variant
- Beam config: beam size 14、move radius 2、top-K path 3
- seed policy: variant 間で同じ query well / seed index の stable SHA256 seed を共有

## 変更点

- `exp211_affine_calibrated_gr_observation_pfbeam` を実装親として `exp213_prefix_structural_prior_pfbeam` を作成。
- helper を `prefix_structural_prior_pfbeam.py` にリネーム。
- affine GR observation variants を外し、raw GR + prefix structural transition variants に絞った。
- structural prior は known prefix tail の `TVT_input + Z` surface を MD に対して robust fit し、expected TVT / expected delta / expected velocity を作る。
- PF は structural expected velocity で初期速度を blend し、各 step で弱い velocity pull と absolute TVT soft prior を使う。
- Beam は last known `TVT_input` start を維持し、absolute TVT soft prior と step-delta cost を足す。
- Beam top-K path を row candidates に保存し、top-K oracle diagnostic、cost gap、path spread を diagnostics に記録する。

## コマンドログ

### 実装時

```bash
make new-steering EXP=exp213_prefix_structural_prior_pfbeam
make new-exp EXP=exp213_prefix_structural_prior_pfbeam SOURCE=experiments/exp211_affine_calibrated_gr_observation_pfbeam
.venv/bin/python -m py_compile experiments/exp213_prefix_structural_prior_pfbeam/prefix_structural_prior_pfbeam.py experiments/exp213_prefix_structural_prior_pfbeam/exp213_prefix_structural_prior_pfbeam_train.py experiments/exp213_prefix_structural_prior_pfbeam/exp213_prefix_structural_prior_pfbeam_inference.py experiments/exp213_prefix_structural_prior_pfbeam/settings.py
.venv/bin/ruff check experiments/exp213_prefix_structural_prior_pfbeam/prefix_structural_prior_pfbeam.py experiments/exp213_prefix_structural_prior_pfbeam/exp213_prefix_structural_prior_pfbeam_train.py experiments/exp213_prefix_structural_prior_pfbeam/exp213_prefix_structural_prior_pfbeam_inference.py experiments/exp213_prefix_structural_prior_pfbeam/settings.py --select F821
.venv/bin/ruff check experiments/exp213_prefix_structural_prior_pfbeam
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp213_prefix_structural_prior_pfbeam/exp213_prefix_structural_prior_pfbeam_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp213_prefix_structural_prior_pfbeam/exp213_prefix_structural_prior_pfbeam_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp213_prefix_structural_prior_pfbeam/exp213_prefix_structural_prior_pfbeam_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp213_prefix_structural_prior_pfbeam/exp213_prefix_structural_prior_pfbeam_inference.py
make validate-exp EXP=exp213_prefix_structural_prior_pfbeam
make prepare-kaggle-notebooks EXP=exp213_prefix_structural_prior_pfbeam EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp213-prefix-structural-prior-pfbeam-train --title 'exp213 prefix structural prior pfbeam train' --run-on-push --strict"
PYTHONPATH=experiments/exp213_prefix_structural_prior_pfbeam .venv/bin/python -c "... synthetic active structural prior PF/Beam smoke ..."
```

結果:

- steering: `docs/legacy/steering/20260707-exp213-prefix-structural-prior-pfbeam`
- experiment: `experiments/exp213_prefix_structural_prior_pfbeam`
- `py_compile`: PASS
- `ruff --select F821`: PASS
- full `ruff check experiments/exp213_prefix_structural_prior_pfbeam`: PASS
- `jupytext --to ipynb` / `--test`: train・inference とも PASS
- `make validate-exp`: PASS
- Kaggle train package: `experiments/exp213_prefix_structural_prior_pfbeam/kaggle/train`
- kernel id: `kentookumura/exp213-prefix-structural-prior-pfbeam-train`
- metadata: CPU / GPU false / internet false / exp072 kernel source あり
- bootstrap manifest: `config.yaml`、`prefix_structural_prior_pfbeam.py`、train/inference `.py`、`settings.py`、`project.yml`、`src/` を含む
- synthetic helper smoke: active structural prior の `run_pf_for_holdout` / `beam_search_for_holdout` が通り、PF output `(8, 10)`、Beam top paths `(3, 10)`、`beam_topk_kept=3`、`structural_prior.active=True` を確認。

## 再現性メモ

- seed policy: `stable_sha256_per_query_well_seed_index_shared_across_structural_variants`
- stochastic components: PF particle propagation / resampling
- CPU/GPU runtime: CPU only、GPU disabled、internet disabled
- gzip 生成物は decompressed content SHA を主証拠として記録する。
- deterministic submission anchor ではない。submission を生成しない。

## 次アクション

1. Jupytext / py_compile / ruff / validate-exp / Kaggle train package 生成を通す。
2. Kaggle train を push する場合は、この note に kernel id / version / runtime / output を追記する。
3. 結果取得後、`KAGGLE_DIRECTION.md` の P0-B backlog を完了扱いで整理するか判断する。

## 2026-07-07 Kaggle train v1 push

ユーザー依頼により Kaggle train を実行する。

実行予定:

- kernel id: `kentookumura/exp213-prefix-structural-prior-pfbeam-train`
- active variant 数: 4 (`raw`, `structural_slope_only`, `structural_weak`, `structural_base`)
- LightGBM config 数: 0
- fold 数: 0
- 合計 booster 数: 0
- GPU: disabled
- internet: disabled
- control / parent 再学習: なし

実行:

```bash
make push-kaggle-train EXP=exp213_prefix_structural_prior_pfbeam
kaggle kernels pull kentookumura/exp213-prefix-structural-prior-pfbeam-train -p /tmp/kaggle-pull/exp213-prefix-structural-prior-pfbeam-train-v1 -m
kaggle kernels logs kentookumura/exp213-prefix-structural-prior-pfbeam-train
kaggle kernels status kentookumura/exp213-prefix-structural-prior-pfbeam-train
timeout 600 kaggle kernels logs -f --interval 30 kentookumura/exp213-prefix-structural-prior-pfbeam-train
```

結果:

- push: success
- kernel version: `1`
- URL: <https://www.kaggle.com/code/kentookumura/exp213-prefix-structural-prior-pfbeam-train>
- pull metadata: success (`/tmp/kaggle-pull/exp213-prefix-structural-prior-pfbeam-train-v1`)
- initial logs: empty while running
- status: `KernelWorkerStatus.RUNNING`
- `logs -f`: 実行中ログなし。ユーザー指示により監視停止。

次アクション: ユーザーから完了連絡を受けたら、同じ kernel id の logs を取得し、必要なら output archive を取得して candidate metrics / PF diagnostics / SHA を `result.md`、`metrics.json`、`experiment_summary.md`、`KAGGLE_DIRECTION.md` に反映する。

## 2026-07-07 Kaggle train v1 完了記録

ユーザーから完了連絡を受け、logs と output を取得した。

実行:

```bash
kaggle kernels status kentookumura/exp213-prefix-structural-prior-pfbeam-train
kaggle kernels logs kentookumura/exp213-prefix-structural-prior-pfbeam-train
kaggle kernels output kentookumura/exp213-prefix-structural-prior-pfbeam-train -p experiments/exp213_prefix_structural_prior_pfbeam/kaggle/output/train_v1
```

結果:

- status: `KernelWorkerStatus.COMPLETE`
- output: `experiments/exp213_prefix_structural_prior_pfbeam/kaggle/output/train_v1`
- runtime: summary 3,415.409 sec / log last 3,632.211 sec
- validation source: exp072 train feature cache 3,783,989 rows / 773 wells
- exp072 cache decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- eval rows: 478,958 rows / 64 wells
- reference candidates present: `pf_ancc`, `pf_z`
- row candidates decompressed SHA: `138e6fb9116630325ebe6bccc136955f472fb787e92c88f6eebdf8f7608ee3b6`
- row candidates raw gzip SHA: `4c9be159f89e5bd0ab619c6ed407110604def56b3df61fa4c53370b0ebf3db0e`

主要 metrics:

- primary baseline `pf_raw_lik_mean`: RMSE 21.081279 / MAE 14.491598 / within10 0.513446
- best non-oracle `exp072_pf_ancc`: RMSE 17.494197 / MAE 10.454963 / within10 0.668491
- `beam_raw_top1`: RMSE 18.339188 / MAE 13.121684 / within10 0.509375
- `beam_structural_base_top1`: RMSE 18.312677 / MAE 13.115793 / within10 0.507662、raw Beam から -0.026510
- `beam_structural_base_top3_oracle`: RMSE 18.287587 / MAE 13.066916 / within10 0.510761
- `pf_structural_weak_lik_mean`: RMSE 28.230909、`pf_raw_lik_mean` から +7.149629
- `pf_structural_base_lik_mean`: RMSE 29.564037、`pf_raw_lik_mean` から +8.482757
- `pf_structural_slope_only_lik_mean`: RMSE 30.621856、`pf_raw_lik_mean` から +9.540576

By-well / bucket:

- `beam_structural_base_top1`: 35/64 wells 改善、29/64 wells 悪化、max regression +15.387904。
- `beam_structural_base_top3_oracle`: 36/64 wells 改善、28/64 wells 悪化、max regression +15.387904。
- Beam structural_base は全 distance bucket で小幅改善し、`1000_plus` は 19.234020 -> 19.205349。
- PF structural variants は 40-41/64 wells で悪化し、max regression は +58 から +59 RMSE。

解釈:

- Beam では structural_base が小さく positive だが、改善幅は RMSE -0.026510 と弱く、既存 `exp072_pf_ancc` に届かない。
- PF では structural prior が大きく悪化し、特に longtail を壊した。
- direct replacement / inference port / submit は行わない。
- P0-C direct generation follow-up は進めず、残す場合は Beam top-K gap / path spread / raw-vs-structural disagreement を confidence feature 材料に限定する。
