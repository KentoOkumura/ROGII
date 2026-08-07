# exp075_compact_tracker_pfbeam_feature_repro_guard

## 状態

- ルート: `ml_model`
- 状態: `inference_v1_completed_submit_check_passed`
- 親: `exp074_compact_tracker_surface_lgbm_candidate_audit`
- feature parent: `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`
- 作成日: 2026-06-16

## 仮説

compact PF/Beam tracker surface は LightGBM の有効な入力候補だが、train feature 生成と model training を同じ notebook に混ぜると後続実験で再利用しにくい。raw train から PF/Beam/likelihood-PF feature を生成する notebook を分離し、LightGBM はその生成物を読むだけにすることで、feature source SHA と model SHA を分けて追跡できる。

## 検証方針

`exp075_compact_tracker_pfbeam_feature_repro_guard_pfbeam_features.ipynb` が raw train から `ravaghi_vs_pixiux_public_replay_tracker_features_train.csv.gz` を 1 回生成する。再現性確認のために同等 CSV を 2 回生成する工程は入れない。

`exp075_compact_tracker_pfbeam_feature_repro_guard_train.ipynb` は生成済み train feature CSV を読み、Pixiux LightGBM 3 configs を 5-fold GroupKFold by `well` で学習する。feature importance は fold/model 単位 CSV、平均 CSV、matplotlib PNG として保存する。

`exp075_compact_tracker_pfbeam_feature_repro_guard_inference.ipynb` は raw test から同じ compact PF/Beam/likelihood-PF feature surface を再生成し、train notebook の saved booster を適用して `submission.csv` を生成する。

保存する主な生成物:

- `ravaghi_vs_pixiux_public_replay_tracker_features_train.csv.gz`
- `compact_tracker_pfbeam_repro_guard_feature_generation_summary.json`
- `compact_tracker_pfbeam_repro_guard_metrics.csv`
- `compact_tracker_pfbeam_repro_guard_feature_importance.csv`
- `compact_tracker_pfbeam_repro_guard_feature_importance_mean.csv`
- `compact_tracker_pfbeam_repro_guard_feature_importance_mean_top.png`
- `compact_tracker_pfbeam_repro_guard_feature_importance_fold_mean_by_fold.csv`
- `compact_tracker_pfbeam_repro_guard_feature_importance_fold_mean.csv`
- `compact_tracker_pfbeam_repro_guard_feature_importance_fold_mean_top.png`
- `compact_tracker_pfbeam_repro_guard_lgb_models/manifest.json`
- `compact_tracker_pfbeam_repro_guard_inference_summary.json`

## 所見

PF/Beam feature generation v4 が stable seed patch 後の CPU notebook で完了。3,783,989 rows / 773 wells / 65 features、decompressed CSV content SHA は `047b80b32e64b595f2a75e7593ecb513e1f27d43de87614dd2de82dae416d5b4`。

LightGBM train v2 は v4 features から完了。`gpu_repro_guard_dp_threads8` の pooled CV RMSE は `lgb_mean=9.699548082062895`。feature importance は raw importance、全fold/model直接平均、fold内model平均後のfold平均を CSV と matplotlib PNG で保存済み。fold平均版の上位は `beam_vs_spatial`, `pf_vs_dense`, `pf_vs_spatial`, `beam_vcons_d`, `pf_z_delta`。

inference v1 は `kentookumura/exp075-compact-pfbeam-lgbm-train` を kernel source として完了。raw-test feature content SHA は `fa82323cff7d24712f109348313a47aaf965c8c8acaa8275f7224d33a35412e8`、submission SHA は `c962cbd1602511c973a7d92b6973c5db790ad7eab310649e025dff411a7c991e`、submit-check は PASS。
