# 設計

## アプローチ

exp072粒子を`(p,r,c)`へ拡張する。cのdrift/transitionはexp364と同じ固定値だが、
本実験のStage 0と判定は独立させる。初期500粒子をc=`-1/0/+1`へ`100/300/100`配分する。
resampling時は各符号へ最低50粒子をsystematic resamplingし、残り350をposterior sign massに
従って配分する。総粒子数と128 seedsは増やさない。

Stage 0は512-row / stride256の3本固定軌道をGRだけで順位付けし、path/score freeze後にtruthを
joinする。top1、MRR、circular control、1000+、hidden-likeを評価する。geometryでrateやcを
予測しない。

実装時の残余規約は次で固定する。512行未満の末尾blockは除外する。negative controlは
strideで列挙した同一well内のGR blockを1 block循環し、complete blockが1本だけなら256行
rollする。score同点時とzero-first reference順位は`0/-1/+1`とする。5 foldの方向PASSは
top1 gain vs zero-first、MRR gain vs zero-first、real minus circular top1がすべて正である
こととする。1000+はblock中央のMD距離が1000 ft以上、1000+とhidden-like 2面の正方向は
GR選択pathのpooled block RMSEがzero pathより小さいこととする。

Stage 1は全gateと別承認時だけ1 treatment PFを走らせる。保存済みexp072 likpf_meanをcontrolとし、
PF controlは再生成しない。scientific failure時はquota、transition、curvature、particle/seed、
blendを調整せず閉じる。

## 実験範囲

- 対象: `exp367_stratified_signed_curvature_pf`
- Route: `pf_beam`
- 親: `exp072_exp063_full_replay_feature_cache`
- 変更: curvature stateとsign-stratified resamplingだけ。
- 固定: 500 particles、128 seeds、noise、momentum、ESS threshold、GR likelihood、mean aggregation。
- Stage 0 gate: top1`>=0.40`、MRR gain`>=0.01`、circular差`>=0.03`、4/5 folds、
  1000+とhidden-like 2面が正方向。
- Stage 1 gate: exp072比`>=0.05 ft`、4/5 folds、1000+/hidden-like/p95回帰`<=0.02 ft`、
  worst`<=0.25 ft`、各符号posterior mass`>=0.02`。

## 再現性設計

- seed: `SHA256(experiment|well|family|seed_index)`からlocal RNGを作る。
- global RNG、thread scheduling依存、well並列でのstream共有は禁止。
- stochastic成分: initial jitter、c transition、propagation、resampling、jitter。
- CPU single worker、GPU off、上限30,600秒。
- raw train / raw testを別生成し、config、input、state diagnostics、predictionのcontent SHAを保存。
- gzipはdecompressed SHAを主証拠にする。

## リスク

- leakage: truthでsign quotaやstateを選ぶ危険。path/score/predictionを先にfreezeする。
- CV/LB不一致: curvature頻度差。
- runtime: exp072と同粒子数でもstate診断で増える。上限超過はfail closed。
- reproducibility: stratified allocationの丸め順を固定する。
- science: weak mode quotaがposterior concentrationを妨げる可能性。

## Stage 0 生成物

- truth-free candidate path bank（gzip、decompressed content SHA）
- truth-free real / circular GR block score bank（gzip、decompressed content SHA）
- raw input identity manifest、freeze manifest
- post-freeze block readout、scope / fold metrics、gate report、summary
- horizontal suffix truthとexp115 hidden-like roleはfreeze SHA readback後だけ読み込む。
