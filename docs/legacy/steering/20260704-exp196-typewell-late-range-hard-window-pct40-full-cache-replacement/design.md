# 設計

## アプローチ

exp192 の hard-window pct50 実装を親にして、generation input の typewell support だけを pct40 に緩める。raw typewell 読み込み直後に元の finite TVT min/max から `typewell_pct` を計算し、`0.40 <= typewell_pct <= 1.00` の rows だけを PF_ANCC、PF_Z、Beam、128-seed likelihood-PF に渡す。

soft prior は無効のままにする。既存 full replay cache は feature generation input には使わず、生成後に exp072 baseline と exp192 pct50 との direct candidate 比較に使う。

## 実験範囲

- 対象実験: `exp196_typewell_late_range_hard_window_pct40_full_cache_replacement`
- Route: `pf_beam`
- 親実験: `exp192_typewell_late_range_hard_window_pct50_full_cache_replacement`
- 変更する変数: hard-window threshold `min_typewell_pct` を `0.50` から `0.40` へ変更し、variant/output 名を pct40 にする。
- 固定する変数: public replay feature builder、PF seeds 128、PF particles 500、n_jobs 8、expected feature count 196、soft prior disabled、LightGBM 0、fold 0、booster 0。
- 比較対象: exp072 original full replay cache と exp192 pct50 full replay cache。

## 再現性設計

- seed policy: stable SHA256 per well / split / feature family。
- stochastic 処理の有無: PF particle propagation、PF resampling、likelihood-PF seed ensemble が stochastic。
- PF/Beam / likelihood-PF / seed bagging の有無: PF_ANCC、PF_Z、Beam、128-seed likelihood-PF を raw files から再生成する。
- 並列処理と乱数の関係: joblib threads を使うが、well 単位で stable seed を導出し、global RNG の消費順に依存しない設計を親実装から維持する。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、`enable_gpu=false`、`enable_internet=false`。GPU 学習はない。
- train cache / test feature regeneration の SHA 記録方針: train cache は raw gzip SHA と decompressed content SHA を分ける。test cache はこの実験では生成しない。
- model manifest / prediction / submission SHA 記録方針: model、prediction、submission を作らないため記録対象外。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --notebook train --strict` 後、metadata と bootstrap/package config が `pct40`、`min_typewell_pct=0.40`、CPU/no internet になっていることを確認する。

## リスク

- リークリスク: train raw TVT は target column として保存するが、PF/Beam generation には使わない。既存 exp072/exp192 cache は生成 input にしない。
- CV/LB 不一致リスク: direct PF/Beam cache の train-side 比較のみで、inference / submit へは進めない。downstream ML へ進める場合も別途 raw-test regeneration parity が必要。
- ランタイム/メモリリスク: exp192 と同程度の CPU long run を想定する。Kaggle output archive の丸ごと取得は必要時のみ。
- 再現性リスク: gzip raw SHA は圧縮条件に依存し得るため、decompressed content SHA を主証拠にする。
