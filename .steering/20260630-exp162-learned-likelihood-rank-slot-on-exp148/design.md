# 設計

## アプローチ

exp148 の full-train learned likelihood add-only runner をベースにする。exp145 feature cache の candidate 別 `learned_prob_*` と `learned_pred_abs_error_*` から learned score を作り、5 候補 (`pf_ancc`, `beam_mean`, `likpf_mean`, `sc_ens`, `hyb`) を row-wise に rank1/rank2/rank3 へ並べる。

rank slot ごとに candidate delta、source code / one-hot、learned prob、predicted error、score、U shape の robust polynomial residual / slope / curvature を作る。slot 間の TVT gap、score/prob/error gap、U disagreement、exp098 heuristic rank1 との source/tvt discrepancy も add-only feature にする。

## 実験範囲

- 対象実験: `exp162_learned_likelihood_rank_slot_on_exp148`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 変更する変数: learned likelihood rank-slot feature group の追加
- 固定する変数: exp072 cache、exp092 U-projection surface、exp145 learned likelihood confidence features、target、GroupKFold seed、LightGBM config family
- 実行 mode: CPU deterministic threads8

## 再現性設計

- seed policy: GroupKFold seed 42、LightGBM deterministic flags、fixed `n_jobs` / `num_threads`
- stochastic 処理の有無: 新規 feature generation には RNG なし。upstream PF/Beam cache と exp145 learned likelihood cache は既存生成物に依存する。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。exp072 / exp145 の保存済み cache を読む。
- 並列処理と乱数の関係: LightGBM CPU deterministic flags と固定 thread count で制御する。
- CPU/GPU runtime と deterministic flags: `runtime.kaggle.enable_gpu=false`、`cpu_deterministic_threads8` のみを active にする。
- train cache / test feature regeneration の SHA 記録方針: gzip は decompressed content SHA を主証拠にし、schema / summary SHA も manifest に残す。
- model manifest / prediction / submission SHA 記録方針: train manifest、fold prediction SHA、pooled prediction SHA を保存する。submission SHA は inference / submit 時にのみ記録する。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks` の strict metadata と notebook JSON validation で確認する。

## リスク

- リークリスク: learned rank は exp145 target-free probability / predicted-error feature だけで作り、true TVT / oracle rank を使わない。
- CV/LB 不一致リスク: exp148 は hidden-safe current-test learned likelihood generation が必要。推論化時は public raw-test cache 固定に戻さない。
- ランタイム/メモリリスク: exp148 より feature 数が増えるため CPU train は長くなる可能性がある。GPU は使わない。
- 再現性リスク: upstream cache の生成自体は過去実験に依存するため、content SHA と manifest を記録する。
