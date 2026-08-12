# 設計

## アプローチ

各 well を exp072 pseudo-tail surface で replayし、T=1 control と gate-on row のみ T=2 に
する treatment を同じ seed base で実行する。各 seed の最初の treatment gate を event とし、
event 後 horizon 内の prediction/error/ESS/resampling を paired 比較する。全 seed prediction は
永続化せず、row-level quantile と event×horizon summary に縮約する。

## 実験範囲

- 対象実験: `exp241_adaptive_likelihood_pf_trajectory_containment_audit`
- Route: `pf_beam`
- 親実験: `exp232_adaptive_robust_likelihood_pf`
- 変更する変数: control を同一コードで再生し、seed-level gate/resampling/ESS と event horizon readoutを追加する。
- 固定する変数: raw GR/typewell GR、prefix、transition、particle/noise、gate threshold、T=2、seed mean。

## 再現性設計

- seed policy: `sha256(exp241, well, paired_likpf)` の base に seed index を加える。
- stochastic 処理の有無: particle initialization、propagation、conditional resampling に乱数を使う。
- PF/Beam / likelihood-PF / seed bagging の有無: likelihood-PF 2 replay、128 seed mean。
- 並列処理と乱数の関係: Numba single worker。well 間並列と global shared RNG は使わない。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU false、internet false。
- train cache / test feature regeneration の SHA 記録方針: input cache と row diagnostics の decompressed content SHA を記録する。test は生成しない。
- model manifest / prediction / submission SHA 記録方針: model/submission なし。row diagnostics と event summary の SHA を記録する。
- Kaggle package bootstrap 確認方針: push 前に generated notebook 内の config、kernel source、CPU/internet metadataを確認する。

## リスク

- リークリスク: event は target-free。target は event確定後のscoreにだけ使用する。
- CV/LB 不一致リスク: train pseudo-tail診断でありLB採用根拠にしない。
- ランタイム/メモリリスク: control+treatment の2 replayで exp232 単体の約2倍。stable hash 4 well shardに分け、seed matrixはwell処理後に縮約して解放する。
- 再現性リスク: conditional resampling 後は乱数消費系列が分岐する。これは faithful replay の診断対象であり、control/treatment が最初に分岐した row を記録する。
