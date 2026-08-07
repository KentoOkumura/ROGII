# exp042_ravaghi_ncc_gr_match_features セッションノート

## 状態

- status: completed
- route: `ml_model`
- parent: `exp041_ravaghi_beam_exact_feature_ablation`
- supporting artifact: `exp029_public_sel15_pf_oof_feature_generation`

## 仮説

Ravaghi notebook の multi-scale NCC (`sc8`, `sc15`, `sc25`, score-weighted
`sc_ens`) と GR/typewell match residual を、raw GR 値の再投入ではなく
confidence / disagreement / offset residual として pseudo-tail single LightGBM
に追加すると、PF/beam が有効な条件の識別に効く可能性がある。

## 実装メモ

- `exp041` を土台に `exp042` を作成。
- `exp029` train well の途中以降を隠した疑似 test rows を入力に使う。
- 各 `(well_id, cutoff_row)` で train horizontal/typewell CSV から Ravaghi NCC/GR match features を再生成する。
- pseudo cutoff 以降の `TVT_input` を隠し、known prefix の `TVT_input` と horizontal `GR`、typewell `TVT/GR` のみを使う。
- feature families:
  - `base_geometry`
  - `base_plus_public_beam_aggregate`
  - `base_plus_ncc_paths`
  - `base_plus_ncc_scores`
  - `base_plus_ncc_disagreement`
  - `base_plus_gr_match_offsets`
  - `base_plus_ncc_gr_match`
  - `base_plus_ncc_gr_match_pf_context`
- `pf_error`、`last_anchor_error`、`beam_error`、`exp026_oof`、exp026 bridge columns は feature から除外。
- direct `beam` / `public_pf_selector` / `pf090_hold010` は report control として残し、model feature candidate とは分ける。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp042_ravaghi_ncc_gr_match_features
uv run python scripts/new_experiment.py --name exp042_ravaghi_ncc_gr_match_features --source experiments/exp041_ravaghi_beam_exact_feature_ablation
uv run ruff check experiments/exp042_ravaghi_ncc_gr_match_features/ravaghi_single_lgbm_audit.py experiments/exp042_ravaghi_ncc_gr_match_features/settings.py
uv run python -m py_compile experiments/exp042_ravaghi_ncc_gr_match_features/ravaghi_single_lgbm_audit.py experiments/exp042_ravaghi_ncc_gr_match_features/settings.py
uv run python scripts/validate_experiment.py --experiment exp042_ravaghi_ncc_gr_match_features
uv run python experiments/exp042_ravaghi_ncc_gr_match_features/ravaghi_single_lgbm_audit.py --max-wells 5 --max-train-rows 1000 --skip-exp026-control --base-estimator HistGradientBoostingRegressor --output-dir /tmp/exp042_smoke
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp042_ravaghi_ncc_gr_match_features --notebook train --kernel-id kentookumura/exp042-ravaghi-ncc-gr-match-train --title "exp042 ravaghi ncc gr match train" --run-on-push --strict
kaggle kernels push -p experiments/exp042_ravaghi_ncc_gr_match_features/kaggle/train
kaggle kernels pull kentookumura/exp042-ravaghi-ncc-gr-match-train -p /tmp/kaggle-pull/exp042-ravaghi-ncc-gr-match-train-check -m
kaggle kernels logs kentookumura/exp042-ravaghi-ncc-gr-match-train
kaggle kernels logs -f --interval 5 kentookumura/exp042-ravaghi-ncc-gr-match-train
kaggle kernels output kentookumura/exp042-ravaghi-ncc-gr-match-train -p /tmp/kaggle-output/exp042_ravaghi_ncc_gr_match_features/train_v1_probe
kaggle kernels logs kentookumura/exp042-ravaghi-ncc-gr-match-train
kaggle kernels output kentookumura/exp042-ravaghi-ncc-gr-match-train -p /tmp/kaggle-output/exp042_ravaghi_ncc_gr_match_features/train_v1
kaggle kernels pull kentookumura/exp042-ravaghi-ncc-gr-match-train -p /tmp/kaggle-pull/exp042-ravaghi-ncc-gr-match-train-complete -m
```

## 結果

- Static checks: PASS
  - `ruff check`: PASS
  - `py_compile`: PASS
  - `scripts/validate_experiment.py --experiment exp042_ravaghi_ncc_gr_match_features`: PASS
- Local smoke: PASS
  - command: `--max-wells 5 --max-train-rows 1000 --skip-exp026-control --base-estimator HistGradientBoostingRegressor`
  - rows: 11,207
  - wells: 5
  - NCC/GR match features generated for 5 `(well_id, cutoff_row)` groups
  - output: `/tmp/exp042_smoke`
  - smoke score is not recorded as CV because it uses 5 wells and a lightweight estimator.
- Kaggle train package: prepared
  - path: `experiments/exp042_ravaghi_ncc_gr_match_features/kaggle/train`
  - kernel id: `kentookumura/exp042-ravaghi-ncc-gr-match-train`
- Kaggle train:
  - pushed version 1 successfully.
  - URL: `https://www.kaggle.com/code/kentookumura/exp042-ravaghi-ncc-gr-match-train`
  - direct `kaggle kernels pull ... -m` succeeds, so the kernel exists on Kaggle.
  - shortly after push, normal `logs`, `logs -f`, and `output` returned empty; treated as API/session output lag or queued/running state rather than a failure.
  - local `logs -f` polling process was stopped after several minutes of no CLI output. Kaggle-side kernel execution was not stopped.
- Full Kaggle audit: completed on Kaggle train notebook version 1.
  - rows: 1,782,279
  - wells: 773
  - runtime from log: about 2,895 seconds
  - output: `/tmp/kaggle-output/exp042_ravaghi_ncc_gr_match_features/train_v1`
  - synced local artifacts:
    - `metrics.json`
    - `artifacts/exp042-ravaghi-ncc-gr-match-train.log`
    - `artifacts/single_lgbm_metrics.csv`
    - `artifacts/single_lgbm_bucket_metrics.csv`
    - `artifacts/single_lgbm_exp026_source_summary.csv`
    - `artifacts/single_lgbm_feature_importance.csv`
    - `artifacts/single_lgbm_split_metrics.csv`
    - `artifacts/single_lgbm_summary.json`
    - `artifacts/single_lgbm_train_summary.csv`
    - `artifacts/single_lgbm_well_metrics.csv`
  - best overall report control: `pf090_hold010`, RMSE 15.089532.
  - strongest overall single-LGBM feature candidate: `base_plus_public_beam_aggregate_bucket_shrink`, original-fold 16.123567 / well-hash 16.132100.
  - best NCC-family bucket-shrink candidate by original-fold: `base_plus_ncc_disagreement_bucket_shrink`, original-fold 17.730920 / well-hash 17.703975.
  - best NCC-family raw candidate by original-fold: `base_plus_ncc_disagreement_raw`, original-fold 17.571472 / well-hash 17.540442.
  - best full NCC/GR match raw candidate by well-hash: `base_plus_ncc_gr_match_raw`, original-fold 17.734118 / well-hash 17.297862.
  - `base_plus_ncc_disagreement_bucket_shrink` vs `base_geometry_bucket_shrink`: -1.358489 original-fold / -1.255343 well-hash.
  - `base_plus_ncc_disagreement_bucket_shrink` vs `public_pf_selector`: +2.558283 original-fold / +2.531339 well-hash.
  - `base_plus_ncc_disagreement_bucket_shrink` vs `pf090_hold010`: +2.641388 original-fold / +2.614443 well-hash.
  - NCC/GR features improve the weak base single-LGBM control but are much weaker than direct PF controls and weaker than exp041 exact beam disagreement.
- LB: not submitted.

## 次のアクション

1. `ravaghi_ncc_gr_match_features` は完了扱いにする。
2. inference port / submit は行わない。NCC/GR match features は direct PF controls と exp041 exact beam disagreement より弱いため、後続では family matrix の診断値として扱う。
