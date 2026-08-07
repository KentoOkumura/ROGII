# 設計

> **閉鎖済み（2026-07-22）**: 本文は旧lineageの設計履歴であり、実装入口ではない。再検証入口はexp209直系の独立実験`exp345_exp209_time_varying_gr_affine_calibration_hmm`とする。

## アプローチ

親HMM mean/stdを凍結し、current-well prefix robust affineを初期state `[b,log a]`とする。outer-train prefixだけでprocess noiseをempirical Bayes推定し、suffix raw GRを`GR_tw(base_mean)`へ対応させるcausal filterを1回実行する。Type Well勾配とbase TVT stdを観測分散へ加え、schedule凍結後にexact HMMを1回だけ再実行する。

## 実験範囲

- 対象: `exp328_time_varying_gr_affine_calibration_hmm`
- Route: `pf_beam`
- 親: `exp308_imputed_gr_confidence_downweight`
- 変更: observation centerの`a_t,b_t` scheduleだけ。
- 固定: sigma/missing weight、transition、grid、momentum、decoder。

## 再現性設計

RNGなし。outer fold/well順、base path、process noise、state schedule、fallback、prediction SHAを保存する。CPU/internet off。32-well runtime gate後だけfull prefix maskを許可する。

## リスク

- base path誤りを校正が吸収する循環: one pass、posterior-std観測分散、prefix mask、worst gateで制限する。
- static affineはexp211/216でtail悪化済み: 本案を最下位優先にし、0.05 ft/4 folds/hard tail gateを要求する。
- runtime: exp295のstate explosionを避けるが、最大1,546 masked HMM runsのため8.5h gateを先に置く。
