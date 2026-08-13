# exp159_spatial_prior_confidence_features_on_exp092 セッションノート

## 2026-06-29 実装

- `backlog/KAGGLE_DIRECTION.md` の `spatial_prior_confidence_features_on_exp092` を実験化した。
- `docs/legacy/steering/20260629-exp159-spatial-prior-confidence-features-on-exp092/` を作成。
- `experiments/exp159_spatial_prior_confidence_features_on_exp092/` を `exp151_tvt_dense_addonly_confidence_features_on_exp092` から作成し、spatial prior confidence 用に差し替えた。
- 親実験は `exp092_u_projection_correction_disagreement_fullrun`、base cache は `exp072_exp063_full_replay_feature_cache`、spatial prior cache は `exp114_spatial_neighbor_prior_signal_audit`。
- 追加特徴は exp114 の `xy_only_k8` / `xy_plus_trajectory_shape_k8` prior value、prior std/count/neighbor quality、PF/Beam/likPF disagreement、exp118 best gate proxy、near/longtail interaction に限定する。
- Spatial prior TVT の direct correction、hard selector、candidate replacement、oracle label、true error rank、target 変更は入れない。

## GPU コストガード

- active variant 数: 1 (`spatial_prior_confidence_addonly`)
- LightGBM config 数: 3 (`lgb0`, `lgb1`, `lgb2`)
- fold 数: 5
- 合計 booster 数: 15
- exp092 control 再学習: なし (`exp092_full_row_control.enabled=false`)
- baseline は保存済み exp092 `lgb1` CV 9.322479896 / Public LB 8.350 を参照する。
- 実行前提: Colab high-memory GPU。大きい cache は DriveFS 直読みではなく `/content` にコピーして実行する。

## 検証ログ

- `python3 -m json.tool experiments/exp159_spatial_prior_confidence_features_on_exp092/exp159_spatial_prior_confidence_features_on_exp092_train.ipynb`: PASS
- `python3 -m json.tool experiments/exp159_spatial_prior_confidence_features_on_exp092/exp159_spatial_prior_confidence_features_on_exp092_inference.ipynb`: PASS
- `python3 -m json.tool experiments/exp159_spatial_prior_confidence_features_on_exp092/exp159_spatial_prior_confidence_features_on_exp092_colab_train.ipynb`: PASS
- `python3 -m py_compile experiments/exp159_spatial_prior_confidence_features_on_exp092/spatial_prior_confidence_features_on_exp092.py experiments/exp159_spatial_prior_confidence_features_on_exp092/settings.py`: PASS
- `make validate-exp EXP=exp159_spatial_prior_confidence_features_on_exp092`: PASS
- `.venv/bin/ruff check experiments/exp159_spatial_prior_confidence_features_on_exp092/spatial_prior_confidence_features_on_exp092.py experiments/exp159_spatial_prior_confidence_features_on_exp092/settings.py`: PASS
- synthetic spatial feature smoke: PASS。200 rows、41 features、feature groups は `spatial_prior_geometry` 4、`spatial_prior_value` 10、`spatial_prior_quality` 16、`spatial_prior_disagreement` 5、`spatial_prior_interaction` 6、finite check PASS。

## Colab runner

- `exp159_spatial_prior_confidence_features_on_exp092_colab_train.ipynb` を作成。
- Drive root は `/content/drive/MyDrive/Kaggle/ROGII`。
- `/content/rogii_cache/exp159_inputs/` に exp072 full replay cache と exp114 spatial OOF gzip をコピーしてから実行する。
- background run は `experiments/exp159_spatial_prior_confidence_features_on_exp092/colab_runs/latest_run.json`、`latest_done_summary.json`、`latest_failed.txt` を使う。
