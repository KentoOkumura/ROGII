# 設計

## アプローチ

exp139 の exp092 + small rank-slot add-only 実装を下敷きにし、rank-slot の採用範囲だけを exp098 full groups に広げる。exp092 の U-projection correction / disagreement は維持し、追加分は `rank_slot_delta`、`rank_slot_identity_score`、`rank_slot_u_projection`、`rank_slot_u_disagreement` をまとめて `rank_slot_feature_groups` で指定する。

Colab では canonical train notebook を直接改変せず、`exp153_full_rank_slot_addonly_on_exp092_colab_train.ipynb` を別に用意する。Drive 上の repo layout を前提に、exp072 cache を `/content/rogii_cache/exp072_artifacts/` へコピーし、その local cache path を training helper に渡す。

## 実験範囲

- 対象実験: `exp153_full_rank_slot_addonly_on_exp092`
- Route: `ml_model`
- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- 変更する変数: exp098 rank-slot feature groups を full add-only で追加するかどうか
- 固定する変数: exp072 cache、target、GroupKFold by well、exp092 U-projection settings、LightGBM family、active mode
- 比較対象: exp092 `lgb1` 9.322479896、exp098 `lgb1` 9.358151052、exp139 `lgb1` 9.324907641、exp147 best `lgb2` 9.397013393

## 再現性設計

- seed policy: GroupKFold seed 42、LightGBM config seed は親系と同じ。新しい PF/Beam RNG は使わない。
- stochastic 処理の有無: 新規特徴生成は deterministic。GPU LightGBM と upstream PF/Beam cache は deterministic anchor としては扱わない。
- PF/Beam / likelihood-PF / seed bagging の有無: 既存 exp072 cache の PF/Beam/likelihood-PF 出力を読むだけで、新規生成はしない。
- 並列処理と乱数の関係: rank-slot feature generation は RNG を使わない。LightGBM は `deterministic=true`、`force_col_wise=true`、`n_jobs=8`、`num_threads=8`。
- CPU/GPU runtime と deterministic flags: primary は GPU double precision `gpu_use_dp=true`。CPU deterministic mode は config に残すが active にはしない。
- train cache / test feature regeneration の SHA 記録方針: gzip は decompressed content SHA を主証拠にする。train summary / model manifest / metrics に入力 path と SHA を残す。
- model manifest / prediction / submission SHA 記録方針: train helper が model SHA、OOF prediction SHA、feature schema、summary を保存する。submission は train-side support 後に別途 inference で記録する。
- Kaggle package bootstrap 確認方針: Kaggle push する場合は `prepare-kaggle-notebooks` 後に metadata と generated package を確認する。今回の primary runtime は Colab runner。

## リスク

- リークリスク: rank-slot score は target-free だが、prefix anchor recovery と raw train cache alignment を検証する。GroupKFold by well を固定する。
- CV/LB 不一致リスク: exp092 Public LB anchor と OOF anchor の両方を参照する。train-side OOF のみで submit しない。
- ランタイム/メモリリスク: full rank-slot union は exp139 より特徴量が増える。Colab high-memory を要求し、DriveFS 直読みを避ける。
- 再現性リスク: Colab GPU / LightGBM GPU は bitwise anchor と呼ばない。log、PID、latest summary、model/prediction SHA で追跡する。
