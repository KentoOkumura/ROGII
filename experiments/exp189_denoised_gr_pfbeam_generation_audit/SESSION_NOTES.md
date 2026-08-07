# exp189_denoised_gr_pfbeam_generation_audit セッションノート

## 目的

`denoised_gr_pfbeam_generation_audit` backlog を実装する。FFT notch や heel calibration には進まず、rolling median / Savitzky-Golay smoothing 単体を PF/Beam observation likelihood に入れて、raw GR likelihood と同一 seed / particles / beam width で比較する。

## 現在の状態

- Route: `pf_beam`
- 状態: `completed_train_side_diagnostic_no_submit`
- CV: diagnostic only
- LB: なし
- 注記: train-side scoped audit のみ。LightGBM 学習、inference port、direct replacement、submit は対象外。

## 実行予定

- active variant 数: GR filter 3 種 (`raw`, `rolling_median_w11`, `savgol_w31_p2`)
- model/config 数: LightGBM なし、PF/Beam generation audit のみ
- fold 数: 0
- 合計 booster 数: 0
- control / 親実験の再学習: なし
- target wells: 最大 64 wells
- score rows: exp072/099 と同じ `TVT_input_missing_equivalent_exp063_rows`
- PF config: 240 particles x 8 seeds / filter
- Beam config: beam size 14、move radius 2
- seed policy: filter 間で同じ query well / seed index の stable SHA256 seed を共有

## コマンドログ

### 実装時

```bash
make new-steering EXP=exp189_denoised_gr_pfbeam_generation_audit
make new-exp EXP=exp189_denoised_gr_pfbeam_generation_audit SOURCE=experiments/exp187_cluster_outlier_alt_typewell_pfbeam_audit
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp189_denoised_gr_pfbeam_generation_audit/exp189_denoised_gr_pfbeam_generation_audit_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp189_denoised_gr_pfbeam_generation_audit/exp189_denoised_gr_pfbeam_generation_audit_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp189_denoised_gr_pfbeam_generation_audit/exp189_denoised_gr_pfbeam_generation_audit_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp189_denoised_gr_pfbeam_generation_audit/exp189_denoised_gr_pfbeam_generation_audit_inference.py
.venv/bin/python -m py_compile experiments/exp189_denoised_gr_pfbeam_generation_audit/denoised_gr_pfbeam_generation_audit.py experiments/exp189_denoised_gr_pfbeam_generation_audit/exp189_denoised_gr_pfbeam_generation_audit_train.py experiments/exp189_denoised_gr_pfbeam_generation_audit/exp189_denoised_gr_pfbeam_generation_audit_inference.py experiments/exp189_denoised_gr_pfbeam_generation_audit/settings.py
.venv/bin/ruff check experiments/exp189_denoised_gr_pfbeam_generation_audit
make validate-exp EXP=exp189_denoised_gr_pfbeam_generation_audit
make prepare-kaggle-notebooks EXP=exp189_denoised_gr_pfbeam_generation_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp189-denoised-gr-pfbeam-audit-train --title 'exp189 denoised gr pfbeam audit train' --run-on-push --strict"
```

結果:

- `py_compile`: PASS
- `ruff check`: PASS
- `jupytext --to ipynb --test`: train / inference とも PASS
- `validate-exp`: PASS
- Kaggle train package: `experiments/exp189_denoised_gr_pfbeam_generation_audit/kaggle/train`
- kernel id: `kentookumura/exp189-denoised-gr-pfbeam-audit-train`
- metadata: GPU false / internet false / run_on_push true / exp072 kernel source あり
- bootstrap manifest: `config.yaml`、`denoised_gr_pfbeam_generation_audit.py`、train/inference `.py`、`settings.py`、`project.yml`、`src/` を含む

### Kaggle train v1

```bash
make push-kaggle-train EXP=exp189_denoised_gr_pfbeam_generation_audit
kaggle kernels pull kentookumura/exp189-denoised-gr-pfbeam-audit-train -p /tmp/kaggle-pull/exp189-denoised-gr-pfbeam-audit-train-v1 -m
kaggle kernels status kentookumura/exp189-denoised-gr-pfbeam-audit-train
kaggle kernels logs kentookumura/exp189-denoised-gr-pfbeam-audit-train
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp189-denoised-gr-pfbeam-audit-train
kaggle kernels output kentookumura/exp189-denoised-gr-pfbeam-audit-train -p experiments/exp189_denoised_gr_pfbeam_generation_audit/kaggle/output/train_v1
```

結果:

- kernel: `kentookumura/exp189-denoised-gr-pfbeam-audit-train` v1
- URL: https://www.kaggle.com/code/kentookumura/exp189-denoised-gr-pfbeam-audit-train
- id_no: 125901169
- status: COMPLETE
- runtime: summary 1,416.547 sec / logs last time 1,539.590 sec
- output: `experiments/exp189_denoised_gr_pfbeam_generation_audit/kaggle/output/train_v1`
- validation source: exp072 train feature cache 3,783,989 rows / 773 wells
- exp072 cache decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- eval rows: 478,958 rows / 64 wells
- row candidates decompressed SHA: `614225c6265a04e30917bc8e417d70d1b6ecab1597816a700786cb99728adc55`
- row candidates raw gzip SHA: `5059ee061d3bef78eaccf3efe2e3ae5b467526fadeaefbfeb7d35365e33fd66f`

主要 metrics:

- primary baseline `pf_raw_lik_mean`: RMSE 20.225464 / MAE 13.027728 / within10 0.564546
- best non-oracle reference `exp072_pf_ancc`: RMSE 17.494197 / MAE 10.454963 / within10 0.668491
- best generated non-oracle `beam_rolling_median_w11_top1`: RMSE 18.028587 / MAE 12.620731 / within10 0.529414
- `beam_savgol_w31_p2_top1`: RMSE 18.136752 / MAE 12.513397 / within10 0.546071
- `beam_raw_top1`: RMSE 18.339188 / MAE 13.121684 / within10 0.509375
- `pf_rolling_median_w11_lik_mean`: RMSE 26.893376 / delta vs raw PF +6.667912
- `pf_savgol_w31_p2_lik_mean`: RMSE 27.943343 / delta vs raw PF +7.717879
- `oracle_best_smoothed_candidate`: RMSE 10.643257 / delta vs primary -9.582207

By-well:

- `beam_rolling_median_w11_top1`: improved 32/64 wells、worsened 32/64 wells、max regression +17.732656。
- `beam_savgol_w31_p2_top1`: improved 35/64 wells、worsened 29/64 wells、max regression +15.722090。
- `pf_rolling_median_w11_lik_mean`: improved 17/64 wells、worsened 47/64 wells、max regression +32.262938。
- `pf_savgol_w31_p2_lik_mean`: improved 17/64 wells、worsened 47/64 wells、max regression +44.016493。
- `oracle_best_smoothed_candidate`: improved 54/64 wells、worsened 10/64 wells、max regression +7.191860。

解釈:

- PF likelihood smoothing は direct candidate として明確に不採用。ESS は少し改善し、resampling rate は低下するが、RMSE は大きく悪化した。
- Beam smoothing は raw Beam より小幅 positive だが、best generated でも `exp072_pf_ancc` に届かず、max well regression も大きいため direct replacement / inference port / submit は行わない。
- smoothed candidate oracle headroom はあるため、残す場合は selector / ML confidence feature 材料に限定する。

## 変更点

- `denoised_gr_pfbeam_generation_audit.py` を追加し、exp072 eval cache を scoring surface として読む scoped PF/Beam audit を実装。
- raw / rolling median / Savitzky-Golay の GR filter contract を `config.yaml` に固定。
- PF は filter 間で同一 stable seed を使い、observation likelihood の GR series だけを変更。
- Beam は同一 beam size / move radius / cost で filtered GR だけを変更。
- candidate metrics、filter delta metrics、bucket/group/by-well metrics、PF diagnostics、target wells、row candidates、summary JSON を保存する設計にした。

## 再現性メモ

- seed policy: `stable_sha256_per_query_well_seed_index_shared_across_gr_filters`
- stochastic components: PF particle propagation / resampling
- CPU/GPU runtime: CPU only、GPU disabled、internet disabled
- gzip 生成物は decompressed content SHA を主証拠として記録する。
- deterministic submission anchor ではない。submission を生成しない。
