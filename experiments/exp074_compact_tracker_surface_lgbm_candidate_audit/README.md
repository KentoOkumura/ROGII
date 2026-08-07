# exp074_compact_tracker_surface_lgbm_candidate_audit

## 状態

- ルート: `ml_model`
- 状態: `kaggle_train_inference_completed_pending_optional_lb`
- 親: `exp070_gpu_reproducibility_guard_for_exp063`
- feature parent: `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`
- 作成日: 2026-06-16

## 仮説

exp070 は exp063 full replay の再現性ガードとしては無効だったが、65-feature compact tracker surface と LightGBM の組み合わせは Public LB 候補として強い可能性がある。目的を再現性ガードから LB 候補監査へ切り替え、ref / kernel version / SHA と CV を分けて記録する。

## 検証方針

train は exp063 が保存した `ravaghi_vs_pixiux_public_replay_tracker_features_train.csv.gz` を固定入力として読み、Pixiux LightGBM 3 configs を 5-fold GroupKFold by `well` で再学習する。特徴量は compact tracker/PF/Beam output frame の 65 features に限定し、target は `TVT - last_known_tvt` とする。

inference は保存済み exp063 test feature CSV を使わず、current raw test から compact PF/Beam/likelihood-PF features を再生成し、exp074 train の保存済み booster を適用して `submission.csv` を生成する。

保存する生成物:

- `compact_tracker_surface_audit_metrics.csv`
- `compact_tracker_surface_audit_by_well.csv`
- `compact_tracker_surface_audit_predictions.csv.gz`
- `compact_tracker_surface_audit_feature_schema.csv`
- `compact_tracker_surface_audit_summary.json`
- `compact_tracker_surface_audit_lgb_models/manifest.json`
- `compact_tracker_surface_audit_inference_summary.json`

## 所見

Kaggle train v1 / inference v1 が完了。`lgb_mean` CV は `9.73150619943287`、submission SHA は `22f9eb3710ccec7741ce8006bee02ed69ed25829439114411cdba0038dcde0bc`。submit-check は PASS。Public LB submit は未実施。
