# 設計

## アプローチ

exp209 の exact HMM smoother を親にし、HMM の observation emission にだけ DTW 補助項を加える。DTW は unknown suffix の horizontal GR と HMM grid 上の typewell GR を constrained DTW で合わせ、anchor TVT、cost、slope、anchor error、stable jitter std/cv、confidence を作る。

HMM 内では direct path を候補として出さず、`emission_ll += alpha * dtw_ll` の形で弱く足す。初期 variants は `alpha=0.05` と `alpha=0.10` の 2 本だけにする。

## 実験範囲

- 対象実験: `exp230_dtw_typewell_warp_hmm_emission_correction`
- Route: `pf_beam`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 変更する変数: DTW 補助 emission の有無と `alpha`
- 固定する変数: exp209 HMM grid / transition / raw GR emission、exp072 saved cache、comparison candidate set、no inference / no submit

## 再現性設計

- seed policy: HMM は RNG なし。DTW jitter は `sha256(experiment, well, "dtw", jitter_idx)` 由来の stable seed。
- stochastic 処理の有無: DTW confidence の安定性診断に deterministic jitter を使う。global RNG は使わない。
- PF/Beam / likelihood-PF / seed bagging の有無: 生成しない。保存済み exp072 cache の `likpf_mean` などを comparison baseline として読むだけ。
- 並列処理と乱数の関係: well 外側並列は `outer_workers=2`。DTW jitter seed は well 固定なので scheduling に依存しない。
- CPU/GPU runtime と deterministic flags: CPU-only、GPU false、internet off。
- train cache / test feature regeneration の SHA 記録方針: train feature gzip SHA と decompressed SHA、schema SHA、summary SHA を記録する。
- model manifest / prediction / submission SHA 記録方針: model / prediction / submission は対象外。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks --strict` で generated package と bootstrap config を確認する。

## リスク

- リークリスク: unknown suffix GR は raw-test でも使えるが、train では評価区間 GR を使うため DTW path 自体が predictor になり得る。直接 candidate 化は禁止。
- CV/LB 不一致リスク: train-side diagnostic のみ。submit 判断には進めない。
- ランタイム/メモリリスク: HMM 2 variants で exp209 HMM より重い。exp072 再生成を切って 12h 以内を狙う。
- 再現性リスク: Numba parallel 浮動小数順序差はあり得るため、SHA と metric は記録する。
