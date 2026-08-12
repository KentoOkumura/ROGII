# exp211_affine_calibrated_gr_observation_pfbeam セッションノート

## 目的

`affine_calibrated_gr_observation_pfbeam` backlog を実装する。prefix affine calibration 済みGRをPF/Beam observation likelihoodへ入れ、raw GR baseline と同一 pseudo-tail surface で比較する。

## 現在の状態

- Route: `pf_beam`
- 状態: `completed_train_side_diagnostic_no_submit`
- CV: train-side diagnostic only
- LB: なし
- 注記: train-side scoped audit のみ。LightGBM 学習、inference port、direct replacement、submit は対象外。

## 実行予定

- active variant 数: 4 (`raw`, `affine`, `raw_structural`, `affine_structural`)
- model/config 数: LightGBM なし、PF/Beam generation audit のみ
- fold 数: 0
- 合計 booster 数: 0
- control / 親実験の再学習: なし
- target wells: 最大 64 wells
- score rows: exp072/099/189 と同じ `TVT_input_missing_equivalent_exp063_rows`
- PF config: 240 particles x 8 seeds / variant
- Beam config: beam size 14、move radius 2
- seed policy: variant 間で同じ query well / seed index の stable SHA256 seed を共有

## コマンドログ

### 実装時

```bash
make new-steering EXP=exp211_affine_calibrated_gr_observation_pfbeam
make new-exp EXP=exp211_affine_calibrated_gr_observation_pfbeam SOURCE=experiments/exp189_denoised_gr_pfbeam_generation_audit
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp211_affine_calibrated_gr_observation_pfbeam/exp211_affine_calibrated_gr_observation_pfbeam_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp211_affine_calibrated_gr_observation_pfbeam/exp211_affine_calibrated_gr_observation_pfbeam_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp211_affine_calibrated_gr_observation_pfbeam/exp211_affine_calibrated_gr_observation_pfbeam_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp211_affine_calibrated_gr_observation_pfbeam/exp211_affine_calibrated_gr_observation_pfbeam_inference.py
.venv/bin/python -m py_compile experiments/exp211_affine_calibrated_gr_observation_pfbeam/affine_calibrated_gr_observation_pfbeam.py experiments/exp211_affine_calibrated_gr_observation_pfbeam/exp211_affine_calibrated_gr_observation_pfbeam_train.py experiments/exp211_affine_calibrated_gr_observation_pfbeam/exp211_affine_calibrated_gr_observation_pfbeam_inference.py experiments/exp211_affine_calibrated_gr_observation_pfbeam/settings.py
.venv/bin/ruff check experiments/exp211_affine_calibrated_gr_observation_pfbeam
make validate-exp EXP=exp211_affine_calibrated_gr_observation_pfbeam
make prepare-kaggle-notebooks EXP=exp211_affine_calibrated_gr_observation_pfbeam EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp211-affine-calibrated-gr-observation-pfbeam-train --title 'exp211 affine calibrated gr observation pfbeam train' --run-on-push --strict"
```

結果:

- steering: `docs/legacy/steering/20260707-exp211-affine-calibrated-gr-observation-pfbeam`
- experiment: `experiments/exp211_affine_calibrated_gr_observation_pfbeam`
- `jupytext --test`: train / inference とも PASS
- `py_compile`: PASS
- `ruff check`: PASS
- `validate-exp`: PASS
- Kaggle train package: `experiments/exp211_affine_calibrated_gr_observation_pfbeam/kaggle/train`
- kernel id: `kentookumura/exp211-affine-calibrated-gr-observation-pfbeam-train`
- metadata: GPU false / internet false / run_on_push true / exp072 kernel source あり
- bootstrap manifest: `config.yaml`、`affine_calibrated_gr_observation_pfbeam.py`、train/inference `.py`、`settings.py`、`project.yml`、`src/` を含む

### Kaggle train v1 push

```bash
make push-kaggle-train EXP=exp211_affine_calibrated_gr_observation_pfbeam
make prepare-kaggle-notebooks EXP=exp211_affine_calibrated_gr_observation_pfbeam EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp211-affine-gr-pfbeam-train --title 'exp211 affine gr pfbeam train' --run-on-push --strict"
make push-kaggle-train EXP=exp211_affine_calibrated_gr_observation_pfbeam
kaggle kernels pull kentookumura/exp211-affine-gr-pfbeam-train -p /tmp/kaggle-pull/exp211-affine-gr-pfbeam-train-v1 -m
kaggle kernels logs kentookumura/exp211-affine-gr-pfbeam-train
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp211-affine-gr-pfbeam-train
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp211-affine-gr-pfbeam-train
```

結果:

- 初回 push は long slug `kentookumura/exp211-affine-calibrated-gr-observation-pfbeam-train` で `SaveKernel` 400。詳細はKaggle CLIに表示されなかった。
- 同じ exp のまま短い slug `kentookumura/exp211-affine-gr-pfbeam-train` / title `exp211 affine gr pfbeam train` に再prepare。
- Kaggle train v1 push 成功。
- URL: https://www.kaggle.com/code/kentookumura/exp211-affine-gr-pfbeam-train
- id_no: 126197987
- metadata: CPU / GPU false / internet false / exp072 kernel source あり。
- 通常 logs と `logs -f` は実行中ログなし。ユーザー指示により一度監視を停止した。その後ユーザー完了連絡を受け、logs / output を取得した。

### Kaggle train v1 completion

```bash
kaggle kernels logs kentookumura/exp211-affine-gr-pfbeam-train
kaggle kernels output kentookumura/exp211-affine-gr-pfbeam-train -p experiments/exp211_affine_calibrated_gr_observation_pfbeam/kaggle/output/train_v1
```

結果:

- kernel: `kentookumura/exp211-affine-gr-pfbeam-train` v1
- URL: https://www.kaggle.com/code/kentookumura/exp211-affine-gr-pfbeam-train
- id_no: 126197987
- status: COMPLETE
- runtime: summary 3,300.654 sec / logs last time 3,543.860 sec
- output: `experiments/exp211_affine_calibrated_gr_observation_pfbeam/kaggle/output/train_v1`
- validation source: exp072 train feature cache 3,783,989 rows / 773 wells
- exp072 cache decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- eval rows: 478,958 rows / 64 wells
- row candidates decompressed SHA: `8dba28dfe8a82536293f56e7f204715679ce2f7354f8e14ac46c6b079ec71465`
- row candidates raw gzip SHA: `9b9e4d8d1b4f98b01d3227388652ae728f425929693786b4c69e5e0726efa22e`

主要 metrics:

- primary baseline `pf_raw_lik_mean`: RMSE 18.640063 / MAE 12.097552 / within10 0.598904
- best non-oracle `exp072_pf_ancc`: RMSE 17.494197 / MAE 10.454963 / within10 0.668491
- `beam_affine_top1`: RMSE 18.065010 / MAE 13.014080 / within10 0.512043、raw Beam 18.339188 から -0.274177
- `beam_affine_structural_top1`: RMSE 18.176860 / within10 0.510565、raw Beam から -0.162328
- `pf_affine_lik_mean`: RMSE 21.184758、`pf_raw_lik_mean` から +2.544695
- `pf_affine_structural_lik_mean`: RMSE 21.143708、`pf_raw_lik_mean` から +2.503645
- `oracle_best_variant_candidate`: RMSE 12.731058 / within10 0.803645
- `oracle_best_nonraw_variant_candidate`: RMSE 13.179088 / within10 0.771594

By-well:

- `beam_affine_top1`: improved 31/64 wells、worsened 33/64 wells、max regression +20.781499。
- `beam_affine_structural_top1`: improved 30/64 wells、worsened 34/64 wells、max regression +20.781561。
- `pf_affine_lik_mean`: improved 26/64 wells、worsened 38/64 wells、max regression +19.521376。
- `pf_affine_structural_lik_mean`: improved 28/64 wells、worsened 36/64 wells、max regression +20.094382。

Affine diagnostics:

- fallback: 0/64 wells
- slope mean / median / min / max: 0.852530 / 0.852239 / 0.506014 / 1.121831
- prefix RMSE mean / median / max: 7.893849 / 7.514946 / 13.680113

解釈:

- affine calibration の prefix fit は安定していたが、PF/likelihood-PF observation としては raw より悪化した。
- Beam affine は raw Beam より小幅 positive だが、既存 `exp072_pf_ancc` に届かず、worst-well regression も大きい。
- direct replacement / inference port / submit は行わない。
- affine quality と raw-vs-affine disagreement は、使うなら selector / confidence feature 材料に限定する。

## 変更点

- `affine_calibrated_gr_observation_pfbeam.py` を正の helper として追加。
- `model.observation_variants` に raw/affine x classic/prefix_structural の 2x2 を定義。
- known prefix only の robust affine fit と fallback guard を実装。
- weak prefix structural prior は `TVT_input + Z` surface を known prefix tail だけでfitし、PF particle weights / Beam path costへ soft cost として入れる。
- PF diagnostics に affine fallback、slope/intercept、prefix RMSE、structural prior diagnostics を出す。

## 再現性メモ

- seed policy: `stable_sha256_per_query_well_seed_index_shared_across_observation_variants`
- stochastic components: PF particle propagation / resampling
- CPU/GPU runtime: CPU only、GPU disabled、internet disabled
- gzip 生成物は decompressed content SHA を主証拠として記録する。
- deterministic submission anchor ではない。submission を生成しない。

## 次アクション

1. `KAGGLE_DIRECTION.md` の P0-A backlog を完了/不採用として外す。
2. P0-C は P0-B 完了後も、P0-A単独が direct PFで悪化したことを前提に慎重に扱う。
3. affine signal を使うなら、`topk_path_confidence_features` などの confidence feature 候補に限定する。
