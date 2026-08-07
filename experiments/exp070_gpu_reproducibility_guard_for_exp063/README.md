# exp070_gpu_reproducibility_guard_for_exp063

## 状態

- ルート: `ml_model`
- 状態: `implemented`
- 親: `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`
- 作成日: 2026-06-13
- 実行: GPU train v4 complete / raw-test-regenerating inference v1 running

## 仮説

exp063 は GPU rerun で OOF SHA と CV が一致しなかった。PF/Beam/likelihood-PF features を exp063 output に固定し、LightGBM の実行モードと再現性向けハイパラだけを変えることで、CV 差分比較に使える再現境界を確認できる。

## 検証方針

`ravaghi_vs_pixiux_public_replay_tracker_features_train.csv.gz` を exp063 から読み、raw PF/Beam generation は行わない。特徴量は exp063 の compact tracker/PF/Beam output frame に限定し、target は `TVT - last_known_tvt` の residual とする。

Inference は hidden/current test rows に対応するため、保存済み exp063 test feature CSV は使わず、exp063 の public replay 実装で raw test files から PF/Beam/likelihood-PF test features を再生成し、exp070 の保存済み LightGBM booster を適用する。

5-fold GroupKFold by `well` で exp063 public Pixiux LightGBM 3 configs を再学習する。比較 mode は `gpu_repro_guard_dp_threads8` と `cpu_deterministic_threads8` を標準にし、必要に応じて `exp063_gpu_float32_reference` を有効化する。

保存する生成物:

- `exp063_repro_guard_metrics.csv`
- `exp063_repro_guard_by_well.csv`
- `exp063_repro_guard_predictions.csv.gz`
- `exp063_repro_guard_feature_schema.csv`
- `exp063_repro_guard_summary.json`
- `exp063_repro_guard_lgb_models/manifest.json`

現在の判断は、GPU train v4 の CV と、raw-test-regenerating inference v1 の LB / submission diagnostics で行う。CPU runtime 比較は別の CPU-only kernel で行い、CPU mode は同じ GPU-enabled session に入れない。

## 所見

v1 は完了したが `well` dtype bug のため主比較では無効。v2/v3 は手動停止。GPU-only train v4 は完了済み。Inference は保存済み exp063 test feature CSV の再利用をやめ、raw test から再生成する package に修正して v1 実行中。
