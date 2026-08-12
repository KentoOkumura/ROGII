# 設計

## アプローチ

exp186 の corrected full replay cache 実装を親実装としてコピーし、soft prior は無効化する。`hard_window_public_replay.py` の raw typewell 読み込み直後に、各 typewell の元の finite TVT min/max から `typewell_pct` を計算し、`0.50 <= typewell_pct <= 1.00` の行だけを PF_ANCC、PF_Z、Beam、128-seed likelihood-PF に渡す。

## 実験範囲

- 対象実験: `exp192_typewell_late_range_hard_window_pct50_full_cache_replacement`
- Route: `pf_beam`
- 親実装: `exp186_typewell_late_range_pfbeam_generation_soft_prior`
- 比較対象: `exp072_exp063_full_replay_feature_cache`
- 変更する変数: raw typewell rows の generation-time support (`typewell_pct >= 0.50`)
- 固定する変数: exp072-style feature schema、PF seeds 128、PF particles 500、joblib per-well parallelism、raw train input、feature count 196、train-cache-only scope

## 再現性設計

- seed policy: `stable_seed("pf_ancc", well)`, `stable_seed("pf_z", well)`, `stable_seed("likpf", split, well)` の stable SHA seed を使う。
- stochastic 処理の有無: PF particle propagation/resampling と likelihood-PF seed ensemble が stochastic。
- PF/Beam / likelihood-PF / seed bagging の有無: PF_ANCC、PF_Z、Beam、128-seed likelihood-PF を生成する。Beam は deterministic。
- 並列処理と乱数の関係: joblib threads で well 単位並列化するが、各 well/feature family の seed は stable key 由来なので thread scheduling に依存しない。
- CPU/GPU runtime と deterministic flags: CPU-only、Kaggle `enable_gpu=false`、internet disabled。
- train cache / test feature regeneration の SHA 記録方針: Kaggle output の raw gzip SHA と decompressed content SHA を分け、decompressed content SHA を主証拠にする。schema SHA と summary SHA も記録する。
- model manifest / prediction / submission SHA 記録方針: この実験では model/prediction/submission を生成しないため not applicable。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` 後、metadata と bootstrap config が hard-window pct50 / CPU-only になっていることを確認する。

## リスク

- リークリスク: typewell filter は raw typewell TVT/GR のみで、train target TVT は feature target column 以外に使わない。比較時の true TVT は評価だけに使う。
- CV/LB 不一致リスク: 初期実装は CV model を学習しない。direct PF/Beam replacement が支持されない場合は downstream 学習へ進めない。
- ランタイム/メモリリスク: exp186 v3 と同程度の multi-hour CPU / large gzip output が想定される。大容量 output は Kaggle CLI 通常 download が OOM する可能性があるため signed URL streaming workaround を使う。
- 再現性リスク: hidden test inference はこの exp では未生成。downstream inference で raw test regeneration parity と SHA を別途記録する必要がある。
