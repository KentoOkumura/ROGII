# 設計

## アプローチ

exp092 の full CV / Public LB とは直接比較せず、exp112 feature cache が存在する shared rows 上で control と add-only variant を同条件比較する。これにより exp112 の 155 wells subset による評価面の違いを分離する。

## 実験範囲

- 対象実験: `exp127_learned_likelihood_features_on_exp092`
- Route: `ml_model`
- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- feature 親: `exp112_learned_pf_likelihood_weight_or_feature_followup`
- 変更する変数: exp112 learned likelihood confidence features の add-only 有無。
- 固定する変数: exp072 base 196 features、exp092 U-projection correction / U-disagreement features、LightGBM config family、GroupKFold by well、residual target。

## Feature Set

追加する exp112 系 feature:

- learned probability / predicted absolute error top1/top2
- top1/top2 margin、entropy、likPF rank、top3 contains likPF
- candidate TVT std/range
- candidate ごとの learned probability、predicted abs error、multiobs score / MAE / NCC
- weighted TVT proxy と candidate TVT は、absolute TVT ではなく `last_known_tvt` と `likpf_mean_tvt` からの差分として追加する

## 再現性設計

- seed policy: fixed GroupKFold seed、LightGBM seed は exp092 config family を継承。
- stochastic 処理の有無: 新規 feature merge に RNG はない。学習は GPU LightGBM の揺れがありうる。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。exp072 / exp112 の保存済み Kaggle output を読む。
- 並列処理と乱数の関係: feature merge は deterministic。LightGBM は `gpu_use_dp=true`、`deterministic=true`、`force_col_wise=true`、`n_jobs=8`、`num_threads=8`。
- CPU/GPU runtime と deterministic flags: 主実行は exp092 と同じ GPU deterministic guard mode。必要なら `cpu_deterministic_threads8` を control として追加できる。
- train cache / test feature regeneration の SHA 記録方針: exp072 / exp112 gzip input は decompressed content SHA を summary に記録する。
- model manifest / prediction / submission SHA 記録方針: model manifest と OOF prediction SHA を記録する。submission は作らない。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` 後に generated package の config / metadata を確認する。

## リスク

- リークリスク: exp112 `fold` を特徴量に使うと split 情報が入るため除外する。valid/test true TVT は feature source に入れない。
- CV/LB 不一致リスク: shared rows は 155 wells subset なので full exp092 CV / LB へ外挿しすぎない。
- ランタイム/メモリリスク: exp092 surface に exp112 feature を追加して 2 variants x 3 LGBM x 5 folds を実行する。subset なので full exp092 より軽い想定。
- 再現性リスク: GPU LightGBM は bitwise deterministic と断定しない。採用候補になった場合は rerun / CPU control / raw-test parity を別途確認する。
