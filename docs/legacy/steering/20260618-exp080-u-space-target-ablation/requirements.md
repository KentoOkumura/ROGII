# 要件

## 依頼

`u_space_target_ablation` を実装する。exp073 deterministic full replay 196-feature surface を固定し、ML target だけを `TVT - last_known_TVT` から `TVT + Z - anchor` 系に差し替えて比較できる Kaggle train notebook を作る。

## 制約

- Route: `ml_model`
- 親実験: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- Feature cache: exp072 の `exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz` を使い、feature set / folds / LightGBM config は exp073 と同一にする。
- Target ablation では valid tail の true TVT を anchor 作成に使わない。`T0` と `Z0` は raw train の既知 prefix 最終行から復元する。
- 再現性: `docs/06_reproducibility.md` に従い、GPU 学習、Kaggle bootstrap、cache/model/prediction SHA の扱いを設計に明記する。

## 受け入れ基準

- Stage 1 として `dTVT`、`dTVT_plus_dZ`、`TVT_plus_Z_abs`、`TVT_plus_Z_minus_T0`、`TVT_plus_Z_minus_T0Z0` を同一 folds / 同一 features / 同一 LightGBM config で比較できる。
- 各 target の予測は TVT 空間に戻して pooled RMSE、well RMSE、distance bucket、tail rank bucket、target 分布を保存する。
- `config.yaml`、train notebook、補助 `.py`、`SESSION_NOTES.md`、`result.md`、`metrics.json` が exp080 用に更新されている。
- `validate_experiment.py`、notebook JSON validation、`py_compile`、`ruff check` が通る。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
