# 設計

## アプローチ

exp092 の U-projection correction / disagreement surface を維持し、PF/Beam 候補 path を well-local normalized U/MD 空間で採点した特徴量を追加する。

`u = candidate_tvt + Z - (T0 + Z0)`、`s = md_since / tail_md_scale` とし、known prefix 末尾の `TVT_input + Z` slope から target-free な prefix extrapolation residual を作る。候補ごとの slope / curvature / roughness、候補間 normalized U disagreement、shape score margin、entropy、instability を LightGBM に add-only で渡す。

## 実験範囲

- 対象実験: `exp130_pfbeam_normalized_diagnostic_score`
- Route: `ml_model`
- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- cache 親: `exp072_exp063_full_replay_feature_cache`
- 変更する変数: PF/Beam normalized diagnostic feature group の追加
- 固定する変数: exp072 feature cache、exp092 U-projection feature surface、target、GroupKFold by well、LightGBM config family

## 再現性設計

- seed policy: GroupKFold / LightGBM seed 固定。新しい PF RNG は追加しない。
- stochastic 処理の有無: 新規 diagnostic feature 生成は deterministic。学習は LightGBM GPU を含む。
- PF/Beam / likelihood-PF / seed bagging の有無: upstream exp072 cache に含まれる PF/Beam/likelihood-PF 候補を固定入力として使う。
- 並列処理と乱数の関係: LightGBM deterministic flags、`n_jobs=8`、`num_threads=8` を config に固定する。
- CPU/GPU runtime と deterministic flags: primary は GPU double precision reproducibility guard、CPU mode も config に残す。
- train cache / test feature regeneration の SHA 記録方針: exp072 cache SHA と feature schema SHA、生成した prediction SHA、model SHA を summary/manifest に記録する。
- model manifest / prediction / submission SHA 記録方針: train manifest と OOF prediction SHA を保存。submission は作らない。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` と `validate-exp` で metadata / bootstrap を確認する。

## リスク

- リークリスク: known prefix の TVT_input だけを使う。validation tail true TVT や candidate error は使わない。
- CV/LB 不一致リスク: PF/Beam diagnostics は public replay cache に依存するため、OOF 改善だけでは submit 判断にしない。
- ランタイム/メモリリスク: exp072 full replay rows 全量に追加特徴量を作るため、Kaggle GPU memory と package size を確認する。
- 再現性リスク: upstream PF/Beam cache の SHA と Kaggle kernel version を記録する。
