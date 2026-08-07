# 設計

## アプローチ

exp092 を親にし、exp072 full replay cache から base 196 features を読み込む。既存 exp092 と同じ U-projection correction / disagreement features を再構成し、その上に normalized shape features だけを add-only で追加する。

exp130 の normalized diagnostic 実装を出発点にするが、`normalized_diagnostic_score`、shape score probability、confidence flag は作らない。LightGBM が無視できる形状特徴だけに限定し、hard selector や target replacement にしない。

## 実験範囲

- 対象実験: `exp149_normalized_shape_addonly_features_on_exp092`
- Route: `ml_model`
- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- 変更する変数: exp092 feature surface に追加する well-local normalized shape feature set
- 固定する変数: exp072 train cache、base 196 features、U-projection settings、target `TVT - last_known_tvt`、GroupKFold by well、LightGBM lgb0/lgb1/lgb2 family、GPU deterministic mode

## 追加特徴

- Geometry: `md_since_norm`、`tail_md_scale`、`u_scale`、`z_rel_norm`、`z_rel_abs_norm`、prefix U slope/roughness norm
- Candidate shape: candidate ごとの `u_norm`、last-anchor drift、prefix-line residual、gradient slope/curvature/roughness、well-local polynomial residual/slope/curvature
- Disagreement: candidate 間 normalized U diff / absdiff、candidate U std/range

## 再現性設計

- seed policy: GroupKFold seed 42。normalized feature generation は RNG なし。
- stochastic 処理の有無: 新規 stochastic feature generation はなし。上流 exp072 PF/Beam cache と GPU LightGBM training は stochastic component として記録する。
- PF/Beam / likelihood-PF / seed bagging の有無: 本実験では再生成せず exp072 cache を読む。raw-test inference port に進む場合は public replay flow 側で再生成 parity を別途確認する。
- 並列処理と乱数の関係: feature generation は deterministic pandas/numpy 処理。LightGBM は deterministic flags、`gpu_use_dp=true`、`force_col_wise=true`、`n_jobs=8`、`num_threads=8` を使う。
- CPU/GPU runtime と deterministic flags: 初回 active mode は `gpu_repro_guard_dp_threads8`。CPU mode は config に保持するが初回 active にはしない。
- train cache / test feature regeneration の SHA 記録方針: exp072 cache SHA、feature schema SHA、train feature schema、prediction SHA、summary JSON を保存する。gzip は decompressed content SHA を主証拠にする。
- model manifest / prediction / submission SHA 記録方針: train manifest に model SHA と model count を保存する。inference / submit に進む場合は prediction SHA と submission SHA を `SESSION_NOTES.md` と `metrics.json` に追記する。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks` 後、generated metadata、GPU/internet flags、kernel source、bootstrap manifest の `config.yaml` と補助 `.py` SHA を確認する。

## GPU Cost Guard

初回 train push 前の予定は、active variant 1、LightGBM config 3、fold 5、合計 booster 15。親 exp092 control 再学習は disabled で、既存 exp092 metrics を baseline として参照する。

## リスク

- リークリスク: `u_scale` や normalized features に true TVT、oracle candidate、absolute error、fold label を入れると漏洩する。known prefix、MD/Z、candidate path のみを使う。
- CV/LB 不一致リスク: exp092 は by-well regression warning がある。global OOF 改善だけでは submit しない。
- ランタイム/メモリリスク: full-row 3.78M rows に shape columns を追加するため、exp130 と同様に memory pressure がある。control を無効化して 1 variant のみにする。
- 再現性リスク: hidden test raw feature regeneration は normal notebook では観測できない。inference port 後は raw-test feature parity と submission-rerun behavior を別途確認する。
