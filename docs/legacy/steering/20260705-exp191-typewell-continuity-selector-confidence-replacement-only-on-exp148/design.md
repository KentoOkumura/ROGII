# 設計

## アプローチ

exp194 の replacement-only train pattern を土台にする。exp148 base surface、U-projection、u disagreement、exp145 learned-likelihood inventory check は維持し、active feature group だけを `exp191_continuity_selector_confidence` に差し替える。

exp191 continuity selector feature block は、exp191 best Viterbi OOF selected path と、exp176 saved boosters から再構築する OOF predicted-error surface で作る。特徴量は selected candidate identity、candidate別 predicted-error/rank、selected predicted-error margin、segment length / boundary / switch、path jump、typewell percentile lower-bound risk に絞る。

## 実験範囲

- 対象実験: `exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 変更する変数: `learned_likelihood_confidence` を外し、`exp191_continuity_selector_confidence` を入れる。
- 固定する変数: base 196 features、projection correction、u disagreement、GroupKFold by well、LightGBM config family、target definition。

## 再現性設計

- seed policy: fixed GroupKFold seed 42。新規 PF/Beam RNG は使わない。
- stochastic 処理の有無: 新規 feature merge は deterministic。LightGBM CPU training は deterministic flags と fixed thread count。
- PF/Beam / likelihood-PF / seed bagging の有無: upstream exp072 / exp099 artifact を読むだけで再生成しない。
- 並列処理と乱数の関係: LightGBM `deterministic=true`, `force_col_wise=true`, `n_jobs/num_threads=8`。
- CPU/GPU runtime: 初回は CPU のみ。GPU path は config に残すが active ではない。
- train cache / test feature regeneration の SHA 記録方針: gzip は decompressed content SHA を主証拠にする。
- model manifest / prediction / submission SHA 記録方針: train split ごとに manifest / prediction SHA を summary と `metrics.json` に記録する。submission SHA は初回なし。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --notebook train_lgb{0,1,2} --strict` で metadata と support ZIP を確認する。

## リスク

- リークリスク: exp176 score reconstruction は OOF fold-held-out saved boosters に限定する。oracle labels は元 fold split 再現にのみ使い、feature columns には入れない。
- CV/LB 不一致リスク: exp160/162 の CV positive / LB negative 前例があるため、global OOF 改善だけで inference / submit しない。
- ランタイム/メモリリスク: exp176 score reconstruction と LGB training を各 split notebook で行うため、Kaggle CPU 時間が長い。lgb0/1/2 を別 kernel に分ける。
- 再現性リスク: 既存 `exp191` 番号が親にも存在するため、summary / notes で duplicate slug の理由を明記する。
