# exp328 セッションノート

## 目的

same-typewell groupに依存しない、current-well causal `a_t,b_t` observation calibrationを低優先で設計する。

## 現在の状態

- 2026-07-21: steering/scaffold作成、設計確定。
- terminal closed / 未実装 / 未実行。
- microbenchmark最大64 matched HMM runs、Stage 0最大1,546、Stage 1最大773 new HMM runs。model/booster 0。

## 固定事項

- base parent pathを1回だけ参照し、causal two-state filter後にscheduleを凍結する。
- process noiseはouter-train prefixからfold-safe empirical Bayes 1回。gridなし。
- group prior、smoother、joint state、2回以上の反復、transition変更、blendは禁止。
- 8.5時間runtime gateとprefix mask/worst guardを必須にする。

## 再現性

RNGなし。fold/well順、base path、process-noise estimate、state schedule、fallback、predictionのcontent SHAを保存する。deterministic submission anchorではない。

## 2026-07-22 閉鎖

親exp308のterminal closeにより本実験も閉鎖した。reparentや実装再開は行わない。causal affineの再検証入口はexp338 chainから独立したexp209直系の`exp345_exp209_time_varying_gr_affine_calibration_hmm`として作成済みで、別steering・別承認とする。

## 次

なし。
