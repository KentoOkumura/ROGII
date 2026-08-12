# 設計

## アプローチ

exp209 exact HMMの各suffix rowで、emission適用前のpredictive rate mean
`mu_pred`と、current-row GR emission適用後のfiltered rate mean`mu_filt`から
rate innovationを計算する。

```text
rate_step = 0.005
u_t = (mu_filt_t - mu_pred_t) / rate_step
C_pos_t = max(0, C_pos_{t-1} + u_t - 0.01)
C_neg_t = max(0, C_neg_{t-1} - u_t - 0.01)
```

refractory外で`C_pos >= 1.0`または`C_neg >= 1.0`になった場合だけtriggerする。
両方が同時に閾値へ達した場合は`C_pos - C_neg`の符号を使い、差が`1e-12`以内なら
triggerしない。trigger方向はpositive / negative rateのどちらか一つで、trigger後は
両CUSUMを0へ戻す。

triggerの次rowから32 transitionsだけ、各source rate stateの既存3-state kernel
`(p_minus, p_stay, p_plus)`へ次を適用する。

```text
gamma = 0.10
p_direction = p_direction + gamma * p_stay
p_stay = (1 - gamma) * p_stay
p_opposite = p_opposite
```

rate grid edgeでdirection側の隣接stateが存在しないsource stateはno-opとする。
新しいrate stateやjump supportは追加しない。activation終了後128 rowsはrefractoryとし、
重複triggerや反対方向への途中切替を許さない。forward passで確定したactivation scheduleを
freezeし、backward passでも同じtime-varying transitionを使う。

これはcurrent emissionの小さい同符号更新を約1 rate cell分まで累積したときだけ
stay probabilityを方向付きで弱める単一変更である。

## 実験範囲

- 対象実験:
  `exp411_predictive_filtered_rate_innovation_destick`
- Route: `pf_beam`
- 科学的親:
  `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 原因証拠:
  `exp408_hmm_message_rate_basin_audit`
- 変更する変数:
  trigger中のrate transition probabilityだけ
- 固定する変数:
  raw入力、prefix/suffix境界、position / rate grid、`sig_r`、`sig_p`、`mom`、
  GR preprocessing / emission、prior、forward-backward、posterior mean
- inference / submission:
  design時点では無効

## Assumption

predictive→filtered rate innovationが、5 ft以上のdatum offset形成より前から、
必要なtrue-rate変化と同符号で累積する。exp408はunder-responseの存在を確立したが、
このtarget-free triggerのlead timeとfalse-trigger率は未確定である。

## Stage 0: fixed32 mechanism preflight

### 対象well

- persistent 16 wells:
  exp408の450 target wellsから、fold 0を4 wells、fold 1--4を各3 wells選ぶ。
  fold内は`sha256("exp411|persistent|" + well)`昇順で固定する。
- matched nonpersistent 16 wells:
  残り323 wellsから同じfold数を選ぶ。persistent wellごとに、suffix row-count quartile、
  raw-GR missing quartile、prefix-row quartileが一致するwellを優先し、
  `sha256("exp411|control|" + well)`でtie breakする。
- sample manifestは実装時にtruth / errorを読む前に固定し、raw / decompressed SHAを保存する。
  persistent membershipはdiagnostic sample選択にだけ使い、decoder / triggerへ渡さない。

### 実行量

- treatment variants: 1
- HMM well-runs: 32
- parent HMM rerun: 0
- model / LightGBM config / trained fold / booster: 0 / 0 / 0 / 0
- PF / Beam / GPU: 0 / 0 / 0
- saved exp209 predictionをread-only controlとする。

### technical AND gate

- selected wells=`16 persistent + 16 control`、重複0、5 foldsを含む。
- trigger / activation schedule freeze前のtruth / error / episode read=0。
- no-trigger synthetic trellisでexp209 posterior mean / log-likelihood差`<=1e-10`。
- zero-active actual wellがある場合、saved exp209 predictionとの差`<=1e-5 ft`。
- posterior row normalization max error`<=1e-5`、finite coverage=1.0。
- activation schedule readback SHA一致。
- active row fractionが`0.001--0.25`、persistent active wells`>=8/16`。
- 773-well保守的runtime projection`<=30,600 sec`、peak RSS`<=25 GB`。

### mechanism AND gate

prediction / trigger scheduleをfreezeした後だけtruthをlate joinする。

- 32-row future true-rate change方向とtrigger方向の一致率`>=0.60`。
- 同一致率が`>0.50`のfoldが`>=4/5`。
- persistent episodeで5 ft onsetより32 rows以上前に初回triggerがあるcoverage`>=0.50`。
- lead-time eligible episodes`>=8`。
- matched control active-row fraction`<=0.10`。
- persistent minus control active-well fraction`>=0.20`。

Stage 0では32-well RMSEをpromotion gateにしない。AND gateを一つでもFAILした場合は、
CUSUM、閾値、gamma、duration、refractoryを救済せずbranchを閉じる。

## Stage 1: full OOF treatment

Stage 0全gate PASSと別のユーザー承認後だけStage 1の実装・Kaggle実行資格を得る。

### 実行量

- treatment variants: 1
- HMM well-runs: 773
- saved parent control HMM rerun: 0
- reporting folds: 5
- model / LightGBM config / trained fold / booster / PF / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0`

### promotion AND gate

- exp209 direct RMSE比`>=0.05 ft`改善。
- 改善fold`>=4/5`。
- persistent episode SSE`>=5%`削減。
- MD 1000+、hidden-like spatial、hidden-like typewell-purgedが各非悪化。
- raw-GR observed / missingが各非悪化。
- by-well RMSE delta p95`<=+0.25 ft`、worst delta`<=+5.0 ft`。
- fixed LikPF / HMM 50:50 blendがsaved parent blendより非悪化。
- active row fraction`0.001--0.25`、全5 foldsでactive well`>0`。
- technical gate、truth-late boundary、SHA readbackを全PASS。

PASSしてもRMSEがexp263 fixed candidateの`8.238331667`を下回らない限り
direct replacementとは呼ばず、target-free candidateとしてのみ後続資格を持つ。
inference / submissionは別承認。

## 再現性設計

- seed policy: RNGなし。well、row、fold、sampleのsortを固定する。
- stochastic処理: なし。
- PF/Beam / likelihood-PF / seed bagging: なし。
- 並列処理: well単位だけ。well内transition、CUSUM、forward/backward順序を固定する。
- runtime: Kaggle private CPU、GPU / internet無効。
- SHA:
  parent input、sample manifest、scientific contract、trigger schedule、
  prediction、metricsを保存し、gzipはdecompressed content SHAを主証拠にする。
- deterministic anchor:
  submissionを生成しないためfalse。数値再現性はprediction content SHAで監査する。
- package:
  実装後のpush前にloose / bootstrap config、Notebook body、asset SHAを照合する。

## リスク

- リークリスク:
  triggerはtarget-freeだが、Stage 0 sample membershipはtruth由来。sample membershipを
  decoderへ渡さず、Stage 1全773 wellsを唯一の性能評価とする。
- CV/LB不一致:
  PF/HMM科学branchであり、Public LB anchorを更新しない。全fold / tail AND gateを優先する。
- false trigger:
  current emissionの微小差を累積するため、系列相関ノイズでtriggerし得る。
- late trigger:
  rate resetは既に形成されたabsolute datum offsetを戻さない。
- runtime:
  trigger schedule保存の追加負荷をStage 0からfull projectionする。
- rescue bias:
  FAIL後の同一OOF threshold / duration / gamma gridは禁止する。

## 2026-07-26 実装時の固定

- ユーザーの実装指示を受け、Stage 0 compact self-contained train候補、
  fail-closed inference候補、fixed32 manifest builder、専用testsを実装した。
- CUSUMは全suffix rowで更新し、active / refractory中は再triggerだけを禁止する。
- trigger rowのtransitionはparentのまま、次rowから32 transitionsをactiveにする。
- future true-rate directionは`(dTVT+dZ)/dMD`のpast 32-row medianと
  future 32-row medianの差で固定した。
- fixed32 manifest SHA256は
  `fbbc62b7cb79e16a7fb436f3a9d11f8975e935ad2475a17e2dec4fd7b142e4d6`。
- 正規Notebook採用、Kaggle package / push / runは実装承認に含めず、別判断のままとした。
