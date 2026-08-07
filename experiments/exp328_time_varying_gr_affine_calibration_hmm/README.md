# exp328_time_varying_gr_affine_calibration_hmm

## 状態

- Route: `pf_beam`
- 状態: 閉鎖済み・未実装・未実行
- 親: `exp308_imputed_gr_confidence_downweight`

親exp308がterminal closeしたため、2026-07-22に本実験も閉鎖した。本仮説はexp338 successor chainへ接続しない。再検証入口はexp209を直接親にした独立の`exp345_exp209_time_varying_gr_affine_calibration_hmm`として作成済みで、本実験自体は再開・reparentしない。

## 仮説

horizontal GRとType Well GRのscale/offsetがsuffix内で緩やかに変わる場合、固定観測校正より、親HMM経路を一度だけ参照して作る因果的`a_t,b_t` scheduleの方がGR emissionを安定化できる。

```text
GR_horizontal,t ≈ a_t × GR_typewell(TVT_t) + b_t
state_t = [b_t, log(a_t)]
```

親HMMのmean/stdを凍結し、Type Well GR at base meanとraw GRから1回だけcausal filterを行う。posterior TVT stdとType Well GR勾配を観測分散に入れ、親経路が不確かな区間ほど校正updateを弱める。schedule凍結後にexact HMMを1回だけ再実行する。

## 既存案との違い

- exp211/216のstatic affine直接適用を繰り返さない。
- exp318のsame-typewell group priorを使わない。
- exp295型のjoint state gridや反復学習を使わない。

## 段階

1. stable SHA 32 wellsでruntime microbenchmark。
2. runtime PASS時だけlast640 prefix maskで親/variant各773、最大1,546 HMM runs。
3. 科学gateもPASSし別承認された場合だけfull suffix新規773 HMM runs。

## 検証方針

- runtime: 32 wells / 64 matched HMM runsからfull auditを外挿し8.5時間以内を必須とする。
- Stage 0: last640 prefix maskで親比0.05 ft以上、4/5 folds、GR NLL改善、boundary jump p95 `<=3 sigma`、hidden-like・worst・fallback guardを全PASSする。
- Stage 1: 保存済みexp308親HMM比0.05 ft以上、4/5 folds、1000+・hidden-like・p95・worst非悪化を要求する。
- affine schedule、process-noise empirical Bayes値、fallbackとcontent SHAをsuffix truth結合前に凍結する。

## 所見

static affineの既存悪化とexp318のgroup prior停止を踏まえ、current-well causal・一回更新へ限定した高リスク設計である。runtimeを科学評価より先に判定し、現在は結果なしとする。

static affineの既存tail悪化があるため最下位優先。本実験の実装、実行、inference、submissionは今後行わない。
