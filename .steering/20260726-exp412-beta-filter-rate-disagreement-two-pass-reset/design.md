# 設計

## アプローチ

first passはexp209 exact HMMを一切変更せず実行し、各suffix rowのfiltered rate mean /
stdとsmoothed rate meanを保存する。

```text
rate_step = 0.005
z_beta_t =
    (mu_smoothed_t - mu_filtered_t) /
    max(sigma_filtered_t, rate_step)
```

各row`t`について直近16 rows（`max(0, t-15):t`）を調べる。

- `abs(z_beta) >= 2.0`のrowが8以上。
- qualifying rowsのsignの75%以上が同じ。

を同時に満たす場合だけrow`t`をactiveとし、多数signをrate方向とする。
positive / negativeが同数、または75%条件を満たさない場合はinactive。
このactive scheduleをtruthを読む前にfreezeし、SHAを固定する。

second passではactive row`t`へ入るrate transitionだけ、first passで決めた方向へ
既存3-state kernelを変更する。

```text
gamma = 0.10
p_direction = p_direction + gamma * p_stay
p_stay = (1 - gamma) * p_stay
p_opposite = p_opposite
```

rate grid edgeでdirection側の隣接stateが存在しないsource stateはno-op。
support、position transition、emission、beta weightは変更しない。active scheduleは
first passから固定され、second passのposteriorで再判定しない。

## 実験範囲

- 対象実験:
  `exp412_beta_filter_rate_disagreement_two_pass_reset`
- Route: `pf_beam`
- 科学的親:
  `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 原因証拠:
  `exp408_hmm_message_rate_basin_audit`
- 先行:
  `exp411_predictive_filtered_rate_innovation_destick`
- 変更する変数:
  frozen beta-filter active rowsのrate transition probabilityだけ
- 固定する変数:
  raw入力、prefix/suffix境界、position / rate grid、`sig_r`、`sig_p`、`mom`、
  GR preprocessing / emission、prior、beta計算、posterior mean
- inference / submission:
  design時点では無効

## Assumption

future betaがforward filterと異なるrate方向を強く持続的に支持する区間では、
absolute position basinが誤っていても、betaのlocal rate方向はtrue rate correctionと
一致する。exp408ではbetaがrate massを回復しながらposition massを悪化させる行が
SSE`38.33%`あった一方、beta自体がwrong basinを作るexclusive backward episodesも
SSE`23.04%`あり、この仮定は高リスクである。

## exp411との順序

- exp412はdesign-onlyで作成する。
- exp411 Stage 0がtrigger support / lead-time不足でFAILした場合、またはexp411 Stage 1が
  future evidence不足を示してFAILした場合だけ、exp412実装資格を得る。
- exp411がStage 1 promotion gateをPASSした場合、exp412は実装せずcloseする。
- ユーザーの明示overrideがある場合だけ、この順序を変更できる。

## Stage 0: fixed32 two-pass mechanism preflight

### 対象well

- backward-cause 8 wells:
  exp408 exclusive `backward_smoothing_reversal` wellsから各fold最低1 wellを
  `sha256("exp412|backward|" + well)`昇順で選び、残り3を未選択全foldから同SHA順で選ぶ。
- forward-cause 8 wells:
  exclusive `forward_transition_prior_hysteresis`から同じ規則で選ぶ。
- matched nonpersistent 16 wells:
  残り323 wellsから、fold、suffix row-count quartile、raw-GR missing quartile、
  prefix-row quartileを一致させ、
  `sha256("exp412|control|" + well)`でtie breakする。
- cause membershipはdiagnostic sample選択とlate readoutだけに使い、first / second passへ渡さない。
- sample manifestは実装時に固定しraw / decompressed SHAを保存する。

### 実行量

- baseline variants / HMM well-runs: `1 / 32`
- treatment variants / HMM well-runs: `1 / 32`
- total HMM well-runs: 64
- model / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
- PF / Beam / GPU: `0 / 0 / 0`
- parent control再実行を含むためKaggle実行は別の明示承認必須。

### technical AND gate

- selected wells=`8 backward + 8 forward + 16 control`、重複0、5 foldsを含む。
- baseline message / trigger schedule / treatment prediction freeze前のtruth / error /
  cause / episode read=0。
- baseline posterior meanとsaved exp209 predictionのmax差`<=1e-5 ft`。
- baseline / treatment normalization max error`<=1e-5`、finite coverage=1.0。
- active scheduleのfreeze / readback SHA一致。
- active row fraction`0.005--0.20`、active wells`>=8/32`。
- 773-well two-pass保守的runtime projection`<=30,600 sec`、peak RSS`<=25 GB`。

### mechanism AND gate

全schedule / predictionをfreezeした後だけtruth / causeをlate joinする。

- active rowの`sign(true_rate - filtered_rate)`とbeta方向一致率`>=0.60`。
- 同一致率が`>0.50`のfoldが`>=4/5`。
- backward-cause wellsのactive-row coverage`>=0.05`。
- backward-cause treatment SSEがbaseline比`>=10%`削減。
- forward-cause treatment SSEの悪化`<=2%`。
- matched control RMSE delta`<=+0.02 ft`、active-row fraction`<=0.10`。

一つでもFAILした場合、threshold、window、8/16、75%、gamma、edge処理、
beta weightを救済せずbranchを閉じる。

## Stage 1: full OOF two-pass treatment

Stage 0全gate PASSと別のユーザー承認後だけ実行資格を得る。

### 実行量

- baseline passes / HMM well-runs: `1 / 773`
- treatment passes / HMM well-runs: `1 / 773`
- total HMM well-runs: 1,546
- reporting folds: 5
- model / LightGBM config / trained fold / booster / PF / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0`

baseline internal messageが既存artifactに無いためparent再実行は不可避だが、
同じfirst-pass predictionがsaved exp209と`<=1e-5 ft`で一致しなければ
treatment評価前にfail-closeする。

### promotion AND gate

- exp209 direct RMSE比`>=0.05 ft`改善。
- 改善fold`>=4/5`。
- exp408 exclusive backward episodes SSE`>=10%`削減。
- exclusive forward episodes SSE悪化`<=2%`。
- MD 1000+、hidden-like spatial、hidden-like typewell-purgedが各非悪化。
- raw-GR observed / missingが各非悪化。
- by-well RMSE delta p95`<=+0.25 ft`、worst delta`<=+5.0 ft`。
- fixed LikPF / HMM 50:50 blendがsaved parent blendより非悪化。
- active row fraction`0.005--0.20`、全5 foldsでactive well`>0`。
- technical、runtime、truth-late、SHA gateを全PASS。

PASSしてもRMSEがexp263 fixed candidateの`8.238331667`を下回らない限り
direct replacementとは呼ばない。inference / submissionは別承認。

## 再現性設計

- seed policy: RNGなし。well、row、fold、sample、pass順を固定する。
- stochastic処理: なし。
- PF/Beam / likelihood-PF / seed bagging: なし。
- 並列処理: well単位だけ。first pass完了・schedule freeze後に同じwellのsecond passへ進む。
- runtime: Kaggle private CPU、GPU / internet無効。
- SHA:
  parent input、sample manifest、first-pass message、active schedule、baseline prediction、
  treatment prediction、metricsを保存し、gzipはdecompressed content SHAを主証拠にする。
- deterministic anchor:
  submissionを生成しないためfalse。baseline / treatment content SHAで数値再現性を監査する。
- package:
  実装後push前にloose / bootstrap config、Notebook body、asset SHAを照合する。

## リスク

- beta誤誘導:
  exclusive backward reversal自体がSSE`23.04%`であり、future evidenceを前方へ移すと
  wrong rateを早期化し得る。
- 二巡目不整合:
  active scheduleはbaseline posteriorから作り、treatment posteriorでは再判定しない。
- リークリスク:
  betaはfuture GRを使うがfuture TVTは使わないためcode competition上許容される。
  truth / cause joinはschedule / prediction freeze後だけ。
- CV/LB不一致:
  高リスクPF/HMM science branchで、LB anchorを更新しない。
- runtime:
  fullは1,546 HMM well-runs。8.5時間projection gateをStage 0で必須とする。
- rescue bias:
  FAIL後のthreshold / window / persistence / gamma gridは禁止する。

## 2026-07-28 実装境界

- 実装承認: あり。
- Stage 0 Kaggle実行承認: なし。
- Stage 1、inference、submission承認: なし。
- fixed32 manifest:
  backward 8 + forward 8 + control 16、SHA256
  `1edb1e1481af84af4e8178fb6e0743fa40315eab0b7441eeff9232b571f93c30`。
- compact self-contained train / fail-closed inference候補を実装する。
- 既存の正規Notebook placeholderは上書きせず、正規採用とpackage作成は別判断とする。
