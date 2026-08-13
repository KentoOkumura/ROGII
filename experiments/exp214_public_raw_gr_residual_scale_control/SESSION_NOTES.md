# exp214_public_raw_gr_residual_scale_control セッションノート

## 目的

`backlog/KAGGLE_DIRECTION.md` の backlog `public_raw_gr_residual_scale_control` を実装する。公開 PF lineage に近い raw GR + known-prefix residual scale を、`exp211/213` と同じ exp072-compatible pseudo-tail 評価面で固定 control として保存する。

これは改善実験ではなく、GRCAL-PFBEAM 系の affine / structural / denoise 変更を測るための物差しである。直接 inference、submit、ML 学習は行わない。

## 現在の状態

- Route: `pf_beam`
- 状態: `completed_train_side_audit`
- CV: train-side diagnostic only、Kaggle train v1 完了
- LB: なし
- 注記: direct inference / submit はしない。exp214 は GRCAL-PFBEAM 系の public-like raw control として使う。

## 実行予定

- active variant 数: 1 (`raw`)
- PF scale 出力: 4 (`3.0`, `5.0`, `8.0`, `12.0`)
- model/config 数: LightGBM なし、PF/Beam generation audit のみ
- fold 数: 0
- 合計 booster 数: 0
- control / 親実験の再学習: なし
- target wells: 最大 64 wells
- score rows: exp211/213 と同じ `TVT_input_missing_equivalent_exp063_rows`
- PF config: 500 particles x 128 seeds / well
- Beam config: raw diagnostic top1、beam size 14、move radius 2
- seed policy: query well + variant + `public_likpf` から stable SHA256 seed base を作り、seed index を加える
- GPU: disabled
- internet: disabled

## 実装メモ

- exp211 を実装親としてコピーし、実験名、config、train/inference notebook、helper 名を exp214 に更新した。
- PF 本体は exp211 の簡易 TVT-state Python loop ではなく、public replay に近い `TVT + Z` surface-state likelihood-PF の numba kernel に差し替えた。
- known prefix の `GR - typewell_GR(TVT_input)` から `gs = clip(std(...), 10, 60)` を計算し、評価 tail では raw horizontal GR と raw typewell GR の likelihood を使う。
- `pf_raw_scale_3`、`pf_raw_scale_5`、`pf_raw_scale_8`、`pf_raw_scale_12`、`pf_raw_lik_mean`、`pf_raw_seed_mean`、`pf_raw_best_seed`、`beam_raw_top1`、oracle diagnostics を row candidates に保存する。
- primary baseline は `pf_raw_scale_5`。

## コマンドログ

```bash
make new-steering EXP=exp214_public_raw_gr_residual_scale_control
make new-exp EXP=exp214_public_raw_gr_residual_scale_control SOURCE=experiments/exp211_affine_calibrated_gr_observation_pfbeam
```

## 検証

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp214_public_raw_gr_residual_scale_control/exp214_public_raw_gr_residual_scale_control_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp214_public_raw_gr_residual_scale_control/exp214_public_raw_gr_residual_scale_control_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp214_public_raw_gr_residual_scale_control/exp214_public_raw_gr_residual_scale_control_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp214_public_raw_gr_residual_scale_control/exp214_public_raw_gr_residual_scale_control_inference.py
.venv/bin/python -m py_compile experiments/exp214_public_raw_gr_residual_scale_control/public_raw_gr_residual_scale_control.py experiments/exp214_public_raw_gr_residual_scale_control/exp214_public_raw_gr_residual_scale_control_train.py experiments/exp214_public_raw_gr_residual_scale_control/exp214_public_raw_gr_residual_scale_control_inference.py experiments/exp214_public_raw_gr_residual_scale_control/settings.py
.venv/bin/ruff check experiments/exp214_public_raw_gr_residual_scale_control --select F821
make validate-exp EXP=exp214_public_raw_gr_residual_scale_control
rg -n "__file__|Path\\(__file__\\)" experiments/exp214_public_raw_gr_residual_scale_control/exp214_public_raw_gr_residual_scale_control_train.py experiments/exp214_public_raw_gr_residual_scale_control/exp214_public_raw_gr_residual_scale_control_inference.py
```

結果:

- Jupytext train / inference 変換: PASS
- Jupytext train / inference `--test`: PASS
- `py_compile`: PASS
- `ruff --select F821`: PASS
- `make validate-exp`: PASS
- train / inference notebook scripts に `__file__` 参照なし

## 再現性メモ

- stochastic components: public-like likelihood-PF particle propagation / resampling
- global RNG: numba kernel 内で per-well stable seed base を `np.random.seed(seed_base + seed_index)` に渡す。並列 RNG は使わない。
- CPU/GPU runtime: CPU only、GPU disabled、internet disabled
- gzip 生成物は decompressed content SHA を主証拠として記録する。
- deterministic submission anchor ではない。submission を生成しない。

## 2026-07-07 Kaggle train v1 push

実行予定は上記の通り、active variant 1、LightGBM config 0、fold 0、booster 0、GPU disabled、internet disabled、control / parent 再学習なし。

実行:

```bash
make prepare-kaggle-notebooks EXP=exp214_public_raw_gr_residual_scale_control EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp214-public-raw-gr-residual-scale-control-train --title 'exp214 public raw gr residual scale control train' --run-on-push --strict"
make push-kaggle-train EXP=exp214_public_raw_gr_residual_scale_control
kaggle kernels pull kentookumura/exp214-public-raw-gr-residual-scale-control-train -p /tmp/kaggle-pull/exp214-public-raw-gr-residual-scale-control-train-v1 -m
kaggle kernels logs kentookumura/exp214-public-raw-gr-residual-scale-control-train
kaggle kernels status kentookumura/exp214-public-raw-gr-residual-scale-control-train
timeout 600 kaggle kernels logs -f --interval 30 kentookumura/exp214-public-raw-gr-residual-scale-control-train
```

結果:

- package: `experiments/exp214_public_raw_gr_residual_scale_control/kaggle/train`
- push: success
- kernel version: `1`
- URL: https://www.kaggle.com/code/kentookumura/exp214-public-raw-gr-residual-scale-control-train
- id_no: `126216224`
- pulled metadata: success (`/tmp/kaggle-pull/exp214-public-raw-gr-residual-scale-control-train-v1`)
- metadata: CPU / GPU false / internet false / competition source `rogii-wellbore-geology-prediction` / kernel source `kentookumura/exp072-exp063-full-replay-feature-cache-train`
- initial logs: empty while running
- status: `KernelWorkerStatus.RUNNING`
- `logs -f`: CLI output は空。ユーザー指示によりローカル監視を停止した。Kaggle 側の kernel 実行は停止していない。

## 2026-07-07 Kaggle train v1 completion

ユーザーから完了連絡を受け、同じ kernel id の status / logs / output を確認した。

```bash
kaggle kernels status kentookumura/exp214-public-raw-gr-residual-scale-control-train
kaggle kernels logs kentookumura/exp214-public-raw-gr-residual-scale-control-train
kaggle kernels output kentookumura/exp214-public-raw-gr-residual-scale-control-train -p experiments/exp214_public_raw_gr_residual_scale_control/kaggle/output/train_v1
```

結果:

- status: `KernelWorkerStatus.COMPLETE`
- output: `experiments/exp214_public_raw_gr_residual_scale_control/kaggle/output/train_v1`
- rows / wells: 478,958 rows / 64 wells
- validation source: exp072 train feature cache 3,783,989 rows / 773 wells
- runtime summary: 3,369.568 sec
- logs last elapsed: 約 3,518 sec
- source gzip SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- source decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- feature schema SHA: `700d38149f583c3ab6574ea7b163c3c8709c2514b675bea381d822f82f4809b8`

Candidate metrics:

| candidate | RMSE | MAE | within10 | delta vs primary |
| --- | ---: | ---: | ---: | ---: |
| `pf_raw_scale_12` | 15.223857 | 9.218137 | 0.673913 | -0.372608 |
| `pf_raw_scale_8` | 15.436026 | 9.356569 | 0.666407 | -0.160439 |
| `pf_raw_scale_5` / `pf_raw_lik_mean` | 15.596465 | 9.503912 | 0.661881 | 0.000000 |
| `pf_raw_scale_3` | 15.676055 | 9.585502 | 0.657717 | +0.079590 |
| `pf_raw_best_seed` | 15.752051 | 9.652642 | 0.656847 | +0.155586 |
| `pf_raw_seed_mean` | 16.029065 | 10.497578 | 0.635546 | +0.432600 |
| `exp072_pf_ancc` | 17.494197 | 10.454963 | 0.668491 | +1.897732 |
| `beam_raw_top1` | 18.339188 | 13.121684 | 0.509375 | +2.742723 |
| `exp072_pf_z` | 24.165177 | 13.864957 | 0.614252 | +8.568712 |

Oracle diagnostics:

- `oracle_best_variant_candidate`: RMSE 11.104328、delta -4.492136、within10 0.839702
- `pf_raw_top3_oracle`: RMSE 14.236926、delta -1.359538、within10 0.731966

PF diagnostics:

- `gr_sigma` mean: 13.897759
- ESS mean: 366.112576
- resampling rate: 0.053373
- `pf_raw_scale_12` は primary `pf_raw_scale_5` に対して `1000_plus` bucket で RMSE 16.403659 -> 16.022127、`500_1000` で 10.807834 -> 10.452503、`250_500` で 7.470886 -> 7.073218 に改善した。

SHA:

- row candidates decompressed: `bef105d23466b13be8d3caee907dd1e5cea1d4f7468907116a95f9bf49344da1`
- row candidates gzip: `198017244c3e094e4f9fdb41b4cfaadf817bab0f21b1a0410832959f85306eda`
- candidate metrics: `e68595c55cc8bc3a935f87086792330df958627e51eb1f209c27f260e58e15f7`
- filter delta metrics: `a80e66ab332ed7d53d3a9ec788acf1befc0fa44e709b6609f3cdc0e6dc2a55c0`
- bucket metrics: `ca80573f5f2e55fe072a07d9dd7ee0b0ad07ea6fc7ff9a3a85481e8a62cc0413`
- by-well: `35e0683ee158c5a36dd1e973f37c3b0a540b9bd629337c4b68c6fa021e6d48c8`
- group metrics: `c760eb23d32c8edc52ff34a72c6fbcdf6ecfd7b5c5a4a99287e925a4efda3b57`
- PF diagnostics: `a7675f68a015a7177b65697ea6ed5b913d6b66a7e00eaaadf59299490cae4643`
- summary: `585f1caf7f17a4c37f04412e25a57147b0dda77ce71f2a0ded2cf39eff886391`
- target wells: `b430641fc305cb56a9332ec4f41ee2497db4bbd269310c969d3754721ad5eac6`
- well status: `0e5d0bcec5677fd0674e6b42d6f5dfc69b09f0f10cb30d7706423e269e11974f`

解釈:

- `public_raw_gr_residual_scale_control` は固定 control として成立した。
- exp211 の軽量 raw control RMSE 18.640063、exp213 の軽量 raw control RMSE 21.081279 より明確に強く、P0-A/P0-B 比較には public-like raw PF control が必要だった。
- best non-oracle は `pf_raw_scale_12` だが、train-side diagnostic のため direct inference / submit はしない。
- oracle headroom は大きいため、scale / seed / top-K path confidence は P2 `topk_path_confidence_features` の材料に限定する。

更新:

- `result.md`、`metrics.json`、`README.md`、`experiment_summary.md`、`backlog/KAGGLE_DIRECTION.md`、steering tasklist を更新した。

## 次アクション

完了。direct inference / submit には進めない。今後の GRCAL-PFBEAM 診断では exp214 `pf_raw_scale_*` を public-like raw control として参照する。
