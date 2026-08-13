---
title: ROGII戦略判断履歴
date: 2026-08-09
types:
  - strategy
experiments: []
topics:
  - history
  - strategy
status: final
summary: "当時のルートKAGGLE_DIRECTION.mdから退避した、2026-08-09までの実験横断の判断履歴。"
---

# ROGII 戦略判断履歴（2026-08-09退避）

- 対応する上位仮説: なし

この文書は、当時ルートにあった`KAGGLE_DIRECTION.md`から退避した判断履歴です。現在の戦略とbacklogは[`backlog/KAGGLE_DIRECTION.md`](../../backlog/KAGGLE_DIRECTION.md)、実験結果は各実験の`result.md`と`metrics.json`を正とします。

## 戦略メモ

### 判断メモ

- 2026-07-30、ユーザーの「実行してください」により、
  `exp487_time_varying_gr_affine_likelihood_pf`のcompact trainをcanonicalへ採用し、
  Kaggle private CPU Stage 0 version 5（id_no `129180524`）を完了した。
  version 1--4は順にraw identity SHA方式、Numba helper、schedule列順、
  saved-control logical provenance判定の実装不整合で停止し、科学条件を変えずに
  修正した。version 5はcausal EKF / bidirectional RTSの2 variants × 32 wells =
  64 PF well-runs、8,192 seed-well trajectories、4,096,000 particle startsを
  `1,191.088 sec`、peak RSS `1.849 GB`で完走した。64/64 variant-wellsの
  truth前freeze、freeze前truth/error/outcome-fold/hidden-role read 0、
  schedule / RTS / covariance / seed / SHA / runtime / RSSを含む全15 technical
  checksはPASSし、Stage 1 technical eligibilityを得た。一方、fixed32の記述
  RMSEはsaved exp404 control `9.616741`に対しcausal `12.634360`
  （`-3.017619 ft`）、RTS `13.391424`（`-3.774684 ft`）で両方大幅悪化した。
  fixed32はCVではないためterminal closeにはしないが、raw GRをschedule updateと
  PF likelihoodへ二重利用する高リスク仮説への強いnegative warningとしてP4へ
  demoteする。Stage 1、inference、submissionは実行せず、別承認待ちとする。
  same-OOF winner、a/b/process-noise/sigma/temperature、well/row gate、
  blend/selector救済は行わない。

- 2026-07-30、ユーザーの「念のためfull wellsに進んでください」により、
  exp440のStage 0 FAIL closedを撤回せず、変更なしcandidateをfull
  773-well OOFで確認した。single-kernel投影が既存9時間guardを超えたため、
  suffix rowsのdeterministic LPTで4 CPU shards
  （`193 / 193 / 193 / 194 wells`）へ分け、candidate HMM 773
  well-runs、saved exp209 control再実行0、model / booster / PF / Beam /
  GPU 0で完走した。strict mergeのtechnical gateは全PASSし、
  freeze前truth/fold/role readも0だった。一方、candidate RMSE
  `12.992063`はparent `11.938287`より`1.053776 ft`悪化、positive fold
  `1/5`、ambiguous-row SSE reduction `-21.3117%`、by-well delta p95 /
  worst `+11.631749 / +45.003490 ft`だった。raw observed / missing、
  高欠損、MD 1000+、hidden-like spatial / typewell-purgedも全て悪化した。
  exp440を`stage1_full_oof_failed_closed`として、blend、selector、
  continuous gate、threshold/lambda変更、same-OOF rescue、rerun、
  inference、submissionなしでterminal closeする。fixed32 pooled改善は
  persistent偏りによるもので、full OOFがこの介入の強いnegative evidenceを
  確定した。

- 2026-07-29、ユーザーの「実行してください」により、exp440のcompact
  train候補を正規train Notebookへ採用し、Kaggle private CPUでfixed32
  Stage 0 version 1（id_no `129064462`）を完了した。実行契約はscientific
  candidate 1本、32 exact-HMM well-runs、5 reporting folds、保存exp209
  control再実行0、LightGBM config / trained fold / booster / fitted model /
  PF / Beam / GPU各0。初回SaveKernel 400はself-contained notebookに不要な
  repository `src/`を含めた1.32 MB packageでkernel作成前に発生し、
  `--no-src`で科学コード・固定asset・親入力を変えず914 KBへ縮小して再検証後、
  canonical version 1のpushに成功した。156,088 rows / 32 wellsを
  `1,464.045 sec`、peak RSS `1.058250 GB`で完走したが、technical gateは
  `13/15`、mechanism gateは`2/8`でFAIL。ambiguity activation
  `18.0526% > 10%`、full runtime projection
  `35,365.847 > 30,600 sec`、predictive-better率
  `42.1180% < 55%`、positive fold `1/5 < 4/5`、persistent改善
  `7/16 wells` / `3/5 folds`、matched-control pooled / p95 delta
  `+0.533492 / +2.582387 ft`が不合格だった。ambiguous-row SSE
  `+12.2001%`とpersistent-episode SSE `+26.3620%`だけはPASSした。
  fixed32 pooledはparent `9.968803`からcandidate `9.280513 ft`へ改善したが、
  persistent選定scopeへの偏りでcontrol safetyを壊しておりCV/promotion
  evidenceにはしない。decisionは
  `stage0_fail_closed_without_ambiguity_lambda_threshold_or_transition_rescue`。
  exp440をbacklogから削除した。その後の明示承認full OOF結果は上記
  2026-07-30メモを正とし、rerun、inference、submission、
  threshold/lambda/transition/blend/well・row gateのsame-OOF救済を行わない。
  exp441--444はrate-transitionの独立仮説であり、exp440救済に再分類しない。

- 2026-07-29、ユーザーの「exp440を実装してください」により、
  `exp440_ambiguity_gated_predictive_prior_hmm`のcompact self-contained
  train/inference候補と専用testを実装した。正規Notebookはscaffoldのまま変更せず、
  Kaggle package / push / Stage 0 run / Stage 1 / inference / submissionは
  未承認のまま保持した。train候補はcandidate自身のcausal predictive jointへ
  親Gaussian emissionを一度適用したprovisional TVT marginalをexp236固定
  peak/valley契約で判定し、raw-GR-observed ambiguous rowだけcandidate
  filtered jointをpredictive jointへ戻す。forward scheduleをbackwardの
  row-wise lambdaへ固定再利用し、missing row、transition、prior、sigma、
  grid、readoutは親exp209を維持する。fixed32 manifestのrole/fold、
  truth、episode、causeは全32 schedule/prediction/diagnostic SHA freeze後に
  late-readする。専用pytest`13 passed`、exp408/411/440関連`39 passed`、
  Jupytext、構文、Ruff F821、strict experiment validationはPASSした。
  これは実装検証だけであり、科学的positive evidence、CV、LBではない。

- 2026-07-30、ユーザーの「実行してください」により、
  `exp446_persistent_tvt_rate_exact_hmm`の正規Notebookを採用し、Kaggle
  private CPU Stage 0 version 1（id_no `129106260`）を1 candidate × 32 wells、
  156,088 suffix rows、保存exp209 parent rerun 0、model / booster / PF / Beam /
  GPU各0で完走した。elapsedは`1,928.728804 sec`、peak RSSは
  `1.131355 GB`。constant-Z parity、dense reference、normalization、
  truth-late、SHA readbackはPASSしたが、773-well runtime投影は
  `46,590.855 > 30,600 sec`でtechnical `17/18`だった。mechanismは`0/7`で、
  zero-directed under-response share削減`-0.061091`、forward / persistent
  episode SSE削減`-0.306441 / -0.214831`、改善`5/16 wells / 2/5 folds`、
  matched-control pooled / p95 delta `+7.159063 / +16.310622 ft`。
  known-Z forcingをrate dynamicsから外すpersistent TVT-rate仮説は、
  狙ったlagを改善せずcontrolを大幅に壊したnegative evidenceと判断した。
  `stage0_fail_closed`としてbacklogから削除し、rate/span/momentum/noise/grid/
  emission/prior/gate/blend/selector救済、rerun、Stage 1、inference、
  submissionを行わない。

- 2026-07-30、ユーザーの「実行してください」により、
  `exp450_dzdmd_conditioned_tvt_rate_likelihood_pf`の正規train Notebookを採用し、
  Kaggle private CPU version 1（id_no `129167787`）を実行した。Stage 0Aは
  sentinel12のparent+exact-transform 24 PF well-runs、3,072 seed-well、
  1,536,000 particle startsを完走したが、exact-coordinate parityをFAILした。
  11 wellsはresampling mismatch 0でも最大`1e-9`級の差が固定`1e-10`を超え、
  `5f4d2a52`では小さい演算差がESS境界を跨いで57 resampling mismatches、
  最大seed prediction差`21.176790850 ft`へ増幅した。artifact readbackもFAIL。
  fail-closedによりStage 0Bの学習型candidateは実行しておらず、科学的な性能結論は
  ない。その後ユーザーは最終temperature-5予測差が微小なら次へ進むことを
  明示承認した。version 2は最終readout`<=1e-6 ft`をhard gate、内部粒子差を
  diagnosticとし、Stage 0A PASS時だけ同一runでStage 0Bへ進む。科学設定と
  mechanism gateは変更していない。version 2はStage 0A PASS・Stage 0B
  candidate 32 wells生成後、exp404 typed source SHAをCSV文字列SHAで照合する
  実装不一致でERROR。expected SHAを変えず親と同一typed SHA関数へ修正し、
  同一canonical kernelのversion 3を`779.671 sec`で完了した。改訂Stage 0Aは
  最終temperature-5差`4.836693e-09 ft`でPASS。Stage 0B fixed32はprefix
  backtest SSE ratio`0.241989`・5/5 foldsをPASSし、persistent pooledも
  `12.785573 -> 12.462589 ft`へ改善したが、狙ったunder-response shareは
  `0.004650`悪化、forward / persistent episode SSE削減は
  `-5.7969% / +1.3603%`、改善foldは`2/5`だった。matched control pooled /
  p95も`+0.292528 / +1.678265 ft`悪化し、全16 gate中6 FAIL。
  `stage0b_mechanism_failed_closed`としてterminal closeし、Stage 1、rerun、
  coefficient/window/PF設定救済、well/row gate、blend/selector、inference、
  submissionを行わない。fixed32全体の`-0.148406 ft`はCVではなく、
  exp450を昇格させない。

- 2026-07-30、exp440の結果を受けたユーザーのpivot依頼により、
  `exp482_isolated_gr_shock_prior_hold`をdesign-onlyで確定した。
  exp440の二峰性・ambiguity schedule・active rowは再利用せず、raw GR現在点が
  `±5`行の前後からrobust z`>=4.5`で孤立し、past predictive meanと
  current-observation leave-one-out meanが`1.05 ft`以内、current emissionだけが
  predictive meanを`1.05 ft`以上動かすAND条件を固定した。active rowだけ
  `normalize(predictive * future beta)`のmeanへ置換し、親exp209 filtered stateと
  次行以降のpredictionは変更しない。Stage A0はraw-only 773-well census、
  Stage A1はtarget-free fixed64のunchanged exp209 message replay 64 wells、
  Stage 1は全gate PASS・別承認時だけ773 message replaysとする。
  scientific candidate 1、candidate state-changing HMM、保存parent prediction
  rerun、LightGBM/model/booster/PF/Beam/GPUは全て0。現時点は実装、正規Notebook
  採用、Kaggle package/run、inference、submission未承認であり、Late phaseの
  低・P3としてexp434と提出直結監査を追い越さない。

- 2026-07-30、ユーザーの「exp482を実装してください」により、
  `exp482_isolated_gr_shock_prior_hold`のcompact self-contained
  train候補、fail-closed inference候補、専用testを実装した。
  raw-only 773-well census、target-free fixed64、unchanged exp209
  message replay、`normalize(alpha_t - emission_t + beta_t)`による
  leave-one-current-observation-out readout、truth-late gate、SHA freezeを
  10章のJupytext sourceへ展開した。parent / 各row LOOは独立exp209 rerunと
  absolute tolerance`5e-7`で一致し、専用pytest`14 passed`、Jupytext、
  構文、Ruff F821/E9、strict experiment validationはPASSした。
  正規Notebookはgeneric scaffoldのまま変更せず、Kaggle package、Stage A0/A1、
  Stage 1、inference、submissionは未承認・未実行。priorityと科学的根拠は
  design-only時から変更しない。

- 2026-07-30、ユーザーの「実行してください」により、
  `exp482_isolated_gr_shock_prior_hold`の正規train Notebookを採用し、
  Kaggle private CPU version 1（id_no `129168015`）でStage A0 raw-only
  census 773 wellsを完了した。isolated shockは17,047 rows、supportは
  763 wellsだった一方、事前固定したzero-shock controlは必要32に対して
  10 wellsしかなく、eligibility FAILとなった。HMM replay、candidate
  prediction、truth/fold join、model / booster / PF / Beam / GPUは全て0。
  raw census SHAは`fdbb653e...02cb`、raw-shock rows decompressed SHAは
  `1615aa35...116`。性能評価前にfixed64 control設計が成立しなかったため
  `stage_a0_eligibility_failed_closed`でterminal closeし、threshold/window/
  control定義の救済、再run、Stage A1、Stage 1、inference、submissionを
  行わない。

- 2026-07-30、ユーザーがzero-shock対照群を不要として次へ進むよう明示したため、
  exp482のterminal履歴を変えず
  `exp488_isolated_gr_shock_prior_hold_support_only`へ分岐した。raw shock、
  message agreement、current-emission conflict、LOO readout、thresholdは
  exp482から不変とし、target-free shock count上位32 wellsだけをKaggle
  private CPU version 2（id_no `129170127`）で評価した。version 1は32-well
  計算後の`numpy.bool_` JSON変換だけでERRORとなり、科学条件を変えず変換と
  testのみを修正した。version 2は183,093行を完走したが、isolated shock
  17,047行・support 763/773 wellsに対して最終AND triggerは
  `0 rows / 0 wells / 0 folds`。candidateは保存exp209 parentと全行同一で、
  RMSEは双方`7.668975975 ft`、改善foldは0/5だった。full runtime投影
  `39,059.748 > 30,600 sec`とsaved-parent replay parityもtechnical FAIL。
  対照群不足ではなく現条件で介入機構が非発火と判断し、
  `stage0_failed_close_without_trigger_threshold_or_output_rescue`で閉じた。
  threshold/window/output救済、Stage 1、inference、submissionは行わない。
  support32はCV / promotion evidenceとして扱わない。

- 2026-07-30、ユーザーの「実行してください」により、
  `exp441_full_support_ou_rate_transition_hmm`の正規train Notebookを採用し、
  Kaggle private CPU Stage 0 version 1（id_no `129095333`）を完走した。
  scientific candidate 1本、32 exact-HMM well-runs、5 reporting folds、
  保存exp209 control再実行0、model / booster / PF / Beam / GPU各0。
  156,088 rowsを`1,582.080 sec`、peak RSS`1.123249 GB`で処理した。
  exact OU mass/moment、dense brute-force、position parity、normalization、
  truth-late、SHA readbackはPASSしたが、full runtime projectionは
  `38,217.120 > 30,600 sec`でtechnical `16/17`。mechanismはcontrol
  pooled / p95 delta `-0.061891 / +0.037121 ft`だけPASSして`2/7`だった。
  zero-directed under-response share削減は`0.022974 < 0.05`、
  forward / persistent episode SSE削減は`-0.001635 / -0.016743`、
  persistent改善は`8/16 wells` / `1/5 folds`でFAIL。全support化は
  control安全性を保ったが主要なpersistent lagを回復せず、
  `stage0_fail_closed`としてexp441をbacklogから削除した。
  Stage 1、rerun、inference、submission、OU parameter / support /
  emission / grid / gateのsame-fixed32救済を行わない。exp442の現行
  Stage 0先行条件も満たさない。

- 2026-07-29、ユーザーの「exp441を実装してください」により、
  `exp441_full_support_ou_rate_transition_hmm`のcompact self-contained
  train/inference候補、exact OU全41-bin CDF kernel、truth-late gate、
  専用15 testsを実装した。これは上記Stage 0実行前の実装履歴であり、
  fixed32結果をCV / LB / route anchor根拠にはしない。

- 2026-07-29、ユーザーの「exp443を実装してください」により、
  `exp443_mean_preserving_trapezoidal_lattice_hmm`のcompact self-contained
  train/inference候補と専用testを実装した。正規Notebookはscaffoldのまま保持し、
  Kaggle package / Stage 0 / Stage 1 / inference / submissionは未承認とした。
  exp439のjoint HMMを親構成に、legal rate marginal、state、GR emission、
  prior、grid、readout、truth-late fixed32評価を維持し、科学差分を固定5-cell、
  trapezoidal mean、`v_eff=max(v_parent,v_lattice_min)`へ限定した。
  exp439 failure edgeの`0.0264 > 0.01500625 ft²`をpositive technical contractとして
  PASSさせ、minimum/effective varianceとinflationをjoint-edge SHAおよび
  variance-floor監査へ含めた。専用pytest`12 passed`、exp439/443関連
  `24 passed`、Jupytext、構文、Ruff F821、strict experiment validationは
  PASSした。これは実装検証であり、fixed32 mechanism evidence、CV、LB、
  route anchor更新根拠ではない。優先度P3とexp441/442先行条件は維持する。

- 2026-07-30、ユーザーの「実行してください」により、exp443の正規train
  Notebookを採用し、Kaggle private CPU Stage 0 version 1
  （id_no `129095370`）を32/32 HMM wellsで完走した。mean / effective variance /
  nonnegative fixed-five support / rate marginal / brute-force / truth-late / SHAは
  PASSし、one-step grid mean biasをほぼ完全に除去した。一方、Stage 1 runtime投影は
  `125406.237 > 30600 sec`でFAIL。mechanismはpersistent改善well `10/16`と
  fold `4/5`だけPASSし、forward-cause SSE削減`5.517%`、persistent SSE削減
  `-5.766%`、control pooled / p95 delta `+0.093698 / +1.394368 ft`はFAILした。
  9,665,508 edgeで有効になったvariance inflationは平均`0.003905 ft²`で、
  mean bias除去だけでは安全なrate-lag改善にならないnegative evidenceと判断した。
  `stage0_fail_closed`としてbacklogから削除し、rerun、Stage 1、inference、
  submission、grid/support/variance/noise/rate/emission/gate/blend/selector救済を
  行わない。

- 2026-07-29、ユーザーの「exp442を実装してください」により、
  `exp442_symmetric_broad_jump_rate_transition_hmm`のcompact self-contained
  train/inference guard候補と専用testを実装した。正規Notebookはscaffoldのまま
  変更せず、Kaggle package / Stage 0 / Stage 1 / inference / submissionは
  未承認とした。exp209 local kernelを99%残し、parent Euler conditional meanを
  中心とする`sigma=0.02`の対称Gaussianを全finite rate-bin Voronoi cellへCDF積分し、
  support外massを捨てるbroad branchを1%だけ厳密混合した。jumpはsamplingせず、
  smoothed transition edgeからbranch responsibility、non-adjacent edge mass、
  signed rate deltaを監査する。専用pytest`12 passed`、local parity / mixture /
  broad mass / brute-force responsibilityは`1e-12`以内、Jupytext、構文、Ruff、
  strict experiment / template validationはPASSした。jump=0の独立exp209
  synthetic prediction差は最大約`0.0011 ft`であり、Stage 0では保存controlとの差に
  含めて判定する。これは実装検証だけで、科学的positive evidence、CV、LB、
  route anchor更新根拠ではない。2026-07-30のユーザー判断により、exp441の
  terminal FAILをnegative contextとして保持しつつ、exp442をexp209に対する
  独立defensive mixture仮説へ再定義した。`0.01` / `0.02`、fixed32、AND gateは
  変更せず、正規train Notebook採用、private CPU package、Stage 0を承認した。
  Kaggle version 1（id_no `129101211`）は1候補×32 HMM wellsを
  `9190.990 sec`、peak RSS `1.191 GiB`で完走した。broad responsibility
  `0.009766954`、non-adjacent mass `0.006845573`、control pooled / p95 delta
  `-0.155414 / +0.069364 ft`はPASSしたが、future direction
  `0.529732 < 0.60`、forward-cause SSE削減`0.2431% < 10%`、
  persistent SSE削減`-4.4385% < +5%`、改善well/fold `9/16` / `2/5`、
  full runtime投影`222019.844 > 30600 sec`をFAILした。technical 14/15、
  mechanism 4/9 PASSで`stage0_fail_closed`とし、backlogから削除した。
  branchは使われたが方向と持続区間を安全に選べないnegative resultであり、
  rerun、Stage 1、inference、submission、weight/sigma/trigger/emission/grid/
  gate救済は行わない。

- 2026-07-29、ユーザー判断によりexp209のrate追従遅れに対する4案を
  design-onlyで確定し、`exp441`--`exp444`へ採番した。当初は全て
  steering / scaffold / config / docsだけだった。その後4案ともcompact候補とtestを
  実装し、exp441/442/443はStage 0、exp444はStage 0Aでterminal FAILした。
  第1案P1 `exp441_full_support_ou_rate_transition_hmm`は、exp209が既に
  destination rate更新後にTVTを進めていることを前提に、rate kernelだけを
  隣接3状態Euler近似から、同じ`momentum=0.998` / `sig_r=0.002`で定まる
  全41-bin exact OU CDF積分へ置換する。新規tuning値なしで1行1binの人工的な
  到達速度制限を除くone-factor候補であり、4案中の最優先とする。
  第2案P2 `exp442_symmetric_broad_jump_rate_transition_hmm`は
  `0.99 * parent + 0.01 * symmetric Gaussian(sigma=0.02)`の1候補だけを固定する。
  exp411のGR innovation方向一致`0.225397` / positive fold`0/5`をnegative
  evidenceとして、方向triggerは使わない。exp441はruntime technical gateと
  主要mechanism gateをFAILしたが、kernel全体を置換するexp441と、exp209 local
  kernelを99%維持するexp442は科学差分が異なる。2026-07-30のユーザー判断により
  exp441を実行前提から外し、exp442を独立仮説としてfixed32 Stage 0で評価した。
  broad branchとcontrol safetyは成立した一方、方向一致、persistent改善、runtimeを
  FAILしたため、same-family救済なしで閉じた。
  第3案P3 `exp443_mean_preserving_trapezoidal_lattice_hmm`は
  `0.5*(r_source+r_destination)*delta_MD-delta_Z`の平均を厳密保存し、
  varianceを`max(parent target, lattice minimum)`とする固定5-cell
  非負projectionである。exp439で`0.01500625 ft^2`のtarget varianceが
  lattice minimum`0.0264 ft^2`を下回ったFAILを緩和するのではなく、
  格子由来variance inflationを明示的な別仮説として保存・監査する。
  rate lagを直接速めないためP3とした。Stage 0では表現contractは成立したが、
  runtimeとpersistent/control safetyをFAILしてbranchを閉じた。
  第4案P4 high-risk `exp444_acceleration_state_exact_hmm`はexp441を構造参照に、
  acceleration `[-0.0005,0,+0.0005] rate/MD-ft`、
  transition`0.08/0.84/0.08`、initial zeroを固定してtrendを持続させる。
  当初のexp441/442先行条件は、2026-07-30のユーザー判断により撤回した。
  exp441のFAILをpositive evidenceやgate救済に使わず、明示trend-memoryが
  full-support OU単体の不足を回復できるかという独立組合せ仮説として実装した。
  state数3倍のためidentity-only hash固定4 wellsのStage 0A runtime/exactnessを先行した。
  Kaggle private CPU version 1（id_no `129154702`）は4 wells / 21,962 rowsを完走し、
  finite、acceleration row-sum、zero-acceleration exp441 OU parity、
  dense posterior、normalization、truth-late、peak RSS `2.282776 GB`はPASSした。
  一方、candidate HMM `746.353694 sec`からのfixed32/full runtime投影は
  `5,970.829552 / 144,232.851372 sec`で固定上限`3,600 / 30,600 sec`をFAILした。
  Stage 0B eligibleはfalse。事前契約どおりstate数/span/transition/kernel/runtime/
  parameter/gate救済、Stage 0B/1、inference、submissionなしでterminal closeした。
  Late phaseのroute anchorは
  ML submitted `exp413` Public LB `7.201`、ensemble `exp082` `7.601`とし、
  exp441--444のnegative resultをanchor supportとは扱わない。

- 2026-07-29、ユーザー判断により「GMM化」ではなく、GRから複数TVTが
  同程度に支持される行だけ現在emissionをneutralizeし、前row posteriorを
  物理transitionで進めたpredictive priorを維持する仮説を
  `exp440_ambiguity_gated_predictive_prior_hmm`としてdesign-onlyで確定した。
  親はexp209、routeは`pf_beam`。GMM / Student-t / Huber、点推定
  `TVT_t=TVT_{t-1}`のhard freeze、soft lambda、threshold gridは使わない。
  raw-GR-observed行のcausal provisional filtered TVT marginalへ
  exp236固定threshold
  (`peak>=0.02`, top2 mass`>=0.10`, top2/top1`>=0.25`,
  separation`>=6 ft`, valley depth`>=0.30`)を適用し、ambiguous行だけ
  emission lambdaを`1.0→0.0`とする。Stage 0はexp411 fixed32の
  1 candidate × 32 HMM well-runs、保存exp209 control再実行0、
  model / booster / PF / Beam / GPU各0。ambiguous rowでpredictive holdが
  provisional updateより良い割合`>=0.55`、改善4/5 folds、ambiguous /
  persistent SSE各`>=5%`削減、persistent改善10/16 wells、control
  pooled / p95安全をAND gateに固定した。全PASS・別承認時だけ
  1 candidate × 773 HMM well-runsのStage 1を許可する。
  exp408ではcurrent emissionの新規wrong反転が`9/807,710 rows`に留まり、
  主因SSE`59.3978%`がemission前のforward hysteresisだったこと、
  exp133 broad ambiguityとexp363 sticky reliabilityがFAILしたことを
  strong negative evidenceとして保持する。このためLate phaseのP3とし、
  exp434 P1とexp436由来fixed-five P3候補より自動昇格させず、P4原因分解より
  前に置く。steering、scaffold、config/docだけを作成し、実装、
  正規Notebook変更、package、run、inference、submissionは未承認である。

- 2026-07-29、`exp437_neighbor_geometry_tvt_only_transition_hmm`をKaggle
  private CPU version 1（id_no `129056603`）で完了した。scientific
  candidate 1本、fixed32 / 156,088 rows / 32 HMM well-runsで、
  parent/control HMM、ML model、LightGBM config、trained fold、booster、
  PF、Beam、GPUは全て0。fold/source、read-time allowlist、truth-late、
  first-difference、transition row-sum、posterior normalization、SHA、
  runtime `39.153270 sec`、peak RSS `0.415966 GB`のtechnical gateは全PASSした。
  一方、candidate RMSE `13.019009088`はexp226 geometry `9.267204778`より
  `3.751804309 ft`悪化した。matched control 16では`0.948324576 ft`改善し、
  保存exp435 dz-onlyにも勝ったが、仮説対象のpersistent 16では
  `6.823650264 ft`悪化。改善foldは`2/5`、paired by-well delta p95 /
  worstは`+21.699228790 / +24.452435654 ft`で、mechanism 7項目中5項目を
  FAILした。exp435の失敗を`-ΔZ`中心だけに帰属する仮説は不支持とし、
  `stage0_fail_closed_without_same_oof_rescue`で閉鎖する。scale / clip /
  noise / emission / grid / subset / gate / blend / selector救済、再実行、
  Stage 1、raw-test再生成、inference、submissionは行わない。完了済みexp437を
  backlogから削除し、exp438 / exp439は独立仮説のまま扱う。

- 2026-07-29、`exp430_huber_seed_evidence_reaggregation`はfixed4 preflight
  version 2、full shard 0--3 version 1、strict merge version 1
  （id_no `129051025`）まで完了した。4 shardは固定1 scientific variant、
  773 PF well-runs、98,944 seed-well trajectories、49,472,000 particle
  startsで、summary SHA、scientific contract SHA、preflight SHA、
  truth-unreadを全てPASS。mergeもinput SHA、3,783,989 rows、773 wells、
  5 folds、finite、shared trajectory identity、weight sum、parent /
  arithmetic parity、truth-lateのtechnical 11 checksを全PASSした。
  一方、Huber RMSE `12.992939553`はmatched trajectory-residual Gaussian
  `12.999103257`を`0.006163704 ft`改善しただけで固定`0.10 ft`に届かなかった。
  nonworse foldは`4/5`だが、shallow、raw-GR-missing、high-missingness、
  roughness-low、hidden-like 2面が悪化。paired-well squared-error delta p95
  `+0.464221656`、worst well `c3957531 +2.658674657 ft`もFAILした。
  保存exp404 parent marginal T=5 `10.914522073`には`2.078417480 ft`、
  arithmetic mean `11.594897884`には`1.398041670 ft`悪化した。
  `huber_seed_evidence_reaggregation_rejected_close_without_rescue`として
  delta / temperature / clip / scale / particle / seed / filtering likelihood、
  well gate、他PF機構とのsame-OOF救済、inference、submissionなしで閉鎖する。
  Huberがmatched Gaussianを僅かに改善しても両者がparent marginal evidenceより
  大幅に弱いため、主因はoutlier感度よりtrajectory-residual evidence objectiveの
  mismatchと解釈する。完了済みexp430をbacklogから削除し、保存済みweight
  concentration / best-seed disagreementだけを読む0-PF原因分解をP4へ追加する。

- 2026-07-28、`exp428_similar_well_gr_registration_map_transfer_readout`を
  canonical Kaggle private CPU version 2（id_no `128932184`）で完了した。
  version 1はGR内部欠損maskをDTW入力にも適用してsupport 0となる実装差を特定し、
  親exp423互換の決定的補間へ修正した。version 2はquery support
  `306 / 773 = 39.586%`で固定70% gateをFAIL。評価可能290 wellsでもrank-1 donor
  global shift MAE `2.529310 ft`はzero `1.105172 ft`より`1.424138 ft`悪化し、
  改善`0/5 folds`。top-5 oracleもzero比`-0.013793 ft`、DTW cost-error Spearman
  `0.075211`、mean ZNCC gain `-0.057438`、local-vs-global block MAE gain
  `-5.050144 ft`で不支持だった。technical/scientific/local gateをすべてFAILし、
  `invalid_or_insufficient_registration_support`として独立rerun、同一OOF救済、
  inference、submission、HMM/PF/Beam統合なしでbranchを閉鎖する。

- 2026-07-28、`exp433_rsd_sparse_anchor_direct_oof_readout`をKaggle private
  CPU version 3（id_no `128939253`）で完了した。exp426 version 1の
  101,231 score rows / 7,787 blocks / 13 offsetsを変更せず、
  unsupported transition-only carryの固定Viterbi 1個を実exp226 OOFへ適用した。
  version 1 / 2のproducer SHA / metrics routing実装欠陥を科学契約不変のまま
  修正し、version 3はtechnical gateを全PASS。prediction SHA
  `c461a14708ffc951060a77e0016a7947f7e2cae1abeb28b539465c0289100377`
  は独立full / probe rerunで一致し、freeze前truth / hidden / episode readは0。
  しかしRMSEは`9.427110→9.692148`で`0.265039 ft`悪化し、改善fold`0/5`、
  1000+ gain `-0.298535 ft`、persistent SSE reduction `-2.797279%`、
  persistent wells改善`160/449`、by-well p95 / worst
  `+3.282839 / +15.926322 ft`だった。near 0--500 ftの小改善を500+の
  carry誤差が上回り、scientific gateは全9条件FAIL。
  `scientific_fail_close_sparse_anchor_branch_without_rescue`として、
  decoder / transition / support / activation / clip / blend / well gate救済、
  inference、submissionなしでRSD sparse-anchor branchを閉鎖する。
  完了済みのためbacklogから削除し、RSD同族の後続は追加しない。優先順位は
  exp413のprediction-only推論は完了済みとし、exp432などdatum-reinjection候補の
  positive evidenceとしてexp433を使わない。

- 2026-07-28、`exp426_rsd_binned_pattern_absolute_reanchor` Stage Aを
  Kaggle private CPU version 1（id_no `128930757`）で実行した。
  `3,783,989 rows / 773 wells / 5 folds / 7,787 blocks`を`164.719113 sec`、
  peak RSS `0.803265 GB`で処理し、inventory、順序、finite score、
  rank / top-3、runtime / memory、fixed-probe parityはPASSした。一方、
  supported blocksは`25.593939% < 95%`、supported wellsは
  `89.262613% < 98%`でtechnical FAIL。truth / hidden-like roleは
  freeze前後とも未読のためscientific評価は行っていない。事前登録どおり
  parameter rescueせずterminal closeし、Stage B / C、inference、
  submissionへ進まない。その後のユーザー判断によるexp433はscore familyを
  救済せず凍結生成物を実OOFへ直接適用する、評価問いの異なる独立readoutとする。
  これはoffset識別精度のnegativeではなく、観測coverageのtechnical FAILとする。

- 2026-07-27、ユーザー依頼によりexp226の保存済みgroup-safe OOF
  `3,783,989 rows / 773 wells / 5 folds`をread-onlyで根本原因監査した。
  global bias `-0.299619 ft`の除去はMSEを`0.1010%`しか説明しない一方、
  exp226互換K16 segment mean offsetを診断上だけ除くとRMSE
  `9.427110→1.130603`、MSE説明`98.5617%`。segment mean errorと前segment
  end errorのPearsonは`0.982951`、境界jump中央値`0.008190 ft`で、
  K16境界不連続ではなく低周波offsetの継承だった。RMSEはsuffix 0--50の
  `1.741257`から2000+の`11.151214`へ成長し、645 persistent episodesは
  rows`18.9943%`でSSE`82.0073%`を占め、onset一行jump中央値は`0.021148 ft`。
  根本機構を`最後の既知TVTを一度だけanchor + spatial donor由来の相対増分を累積 +
  suffix内absolute re-anchorなし`と確定する。target local structureとの小さな
  signed rate mismatchが積分され後続segmentへvertical offsetとして持ち越される。
  geometry / pre-U / final RMSEは`10.077950 / 9.500816 / 9.427110`でGR/Uは
  pooled・5/5 folds改善のため単独原因ではないが一部threshold triggerにはなる。
  donor max下位/上位quartileのwell RMSE中央値は`4.099483 / 7.774613`、
  episode well率`43.52% / 72.68%`で、遠いdonorと長いsuffixを増幅条件とする。
  公開deterministic v6とportのgeometry、coeff、local-linear、columns、GR、Uを含む
  9数値核は固定synthetic入力で最大差`0.0`。global calibration、誤anchor、行順、
  特定fold、K16境界jump、K=16単独、v6 port bugを棄却した。外部weight依存v7/v8の
  learned residual layer不在はv6の性能上限だが移植不具合ではない。truth利用oracleは
  deployable correctionではなく、exp285/281/333のnegative tail guardを維持し、
  本監査だけを根拠にoffset値のhard correctionやsame-OOF rescueを追加しない。
  詳細は`docs/surveys/exp226_offset_root_cause_audit_20260727.md`へ集約した。

- 2026-07-26、ユーザー判断により `scale5_likpf_full_replacement_on_exp335` を
  `exp413_scale5_likpf_full_replacement_on_exp335` として設計確定し、その後の
  明示依頼でtrain-side実装候補まで完成した。
  exp404 の `scale 5・gs×1.0` RMSE `10.914522073` は同じ trajectory の
  arithmetic mean `11.594897884` より `0.680375810 ft` 良く、現行 Public-LB
  reference exp335 `7.517` へ伝播させる独立根拠として扱う。13候補目は追加せず、
  fixed12 の `likpf_mean` slot と4派生 formula の計5 slotを scale5 x1.0 へ置換し、
  fixed7 slotを維持する。clean273、selector88→compact74、signed23、final370を
  全再生成し、予定量を40 CPU selector + 20 CPU signed selector + 15 GPU
  downstream = 75 boosters、saved exp335 control再学習0、train新規PF0に固定した。
  primary gateはsaved exp335比 `>=0.03 ft`、3/5 folds nonworse、near / mid / 1000+ と
  hidden-like 2面各 `<=+0.02 ft`。by-well p95 / worst / +1 / +3 / +5悪化well数は必須
  report-onlyとし、LB-oriented判断とtrain-side robust promotionを分離する。
  exp404 x1.3 FAILは維持し、same-OOF scale / multiplier / feature / candidate / weight救済は
  禁止する。frozen inputの4種SHA、replacement overlay、5 changed / 7 unchanged、
  full clean273再構築、Stage C/S/D、fixed gateをhelperと722行・9節のJupytext
  train候補へ実装した。専用8 testsと親exp264/335回帰25 tests、
  Jupytext/py_compile/Ruff/strict experiment validationはPASS。2026-07-27の
  Stage 0 version 3は3,783,989 rows / 773 wells / 5 partitionsでtechnical PASSし、
  5 changed / 7 unchanged、formula / old-mean parity max差0、model / PF 0を確認した。
  version 1/2はfloat32保存精度と監査算術dtypeの不一致で停止したが、許容幅を緩めず
  cache精度と親contract演算順序のexact parityへ修正した。続くStage C version 3は
  `6378.321 sec`、40/40 CPU models、25 compact partitions / 18,919,945 rows、
  45,407,868 outer-valid score rowsを完了した。expected-error MAE
  `3.720634 vs prior 5.700200`、within10 logloss `0.349579 vs 0.499814`、
  Brier `0.108064 vs 0.160703`で各5/5 folds改善し、score / leakage guardをPASSした。
  version 1は最終reproducibility manifest seed欠落、version 2は学習前のNotebook
  埋込run flag不一致で停止し、科学条件を変えずseed作成と埋込bootstrap実展開監査を
  加えてversion 3を完了した。2026-07-28のStage S version 1は`2984.194 sec`、
  20/20 CPU models、25 signed partitions / 18,919,945 rows、45,407,868 score rowsを
  完了した。pooled signed-residual RMSEは`8.291963 vs prior 10.854996`、
  5/5 folds改善でtechnical / score gateをPASSした。12候補中11候補は改善したが、
  `exp226_w500_50_50`だけはprior比`0.123613 ft`悪化した。Stage D version 2は
  15/15 GPU modelsを完了し、final TVT RMSE `7.884802794`、saved exp335比
  `0.261304961 ft`改善、5/5 foldsと全5 scopes改善でprimary gateをPASSした。
  by-well p95 / worstは`+1.228715 / +9.033462 ft`のreport-only悪化だった。
  2026-07-29のcurrent-test CPU inference version 3は保存済み40/20/15 modelsを
  新規booster 0で適用し、14,151 rows / 3 wellsのrow/order/finite/SHA監査をPASSした。
  scale5はarithmetic meanから14,093 rows変化し、abs/delta parityは0.0 ft、
  prediction decompressed SHAは
  `875a1334ae3c90f841414f8f98d8877fb06234e17e0fd0b8d46385170a584dc4`。
  Kaggle Notebook自身がsample互換submissionを生成し、取得後submit-checkをPASSした。
  submission SHAは
  `e9bb6bca7e19a087997c1f8d1d708d8ba0af21e770f5e44e1f1a52078142772f`。
  ユーザー実施code submission ref `55078306`はversion 3の公開test固定row/well
  assertによりhidden rerun errorとなった。固定assertだけをsample由来の動的
  row / ID / nonempty-well契約へ置換したversion 4は`432.680 sec`でCOMPLETEし、
  version 3公開出力と完全一致、submit-checkもPASSした。ユーザー実施のversion 4
  code submission ref `55080377`はPublic LB `7.201`でCOMPLETEし、exp335
  `7.517`比`-0.316`でML Public-LB referenceを更新した。正規Notebook編集は
  未実行で、train-side robust promotionはtail readoutに基づき分離を維持する。

- 2026-07-26、exp209 persistent-offsetの未保存内部messageを直接分離する
  `exp408_hmm_message_rate_basin_audit`をKaggle private CPU version 3
  （id_no `128636642`）で完了した。current exp209 HMM 1 variantだけを450 / 450 wells、
  2,264,135 suffix rowsへ再生し、638 episodes / 807,710 rowsのpredictive prior、
  filtered alpha、smoothed posterior、backward beta、rate mass、log-sum / max-product差を
  stream保存した。runtime`15,930.997 sec`、peak RSS`3.588 GB`、exp270 posterior mean
  parity最大差`0.0 ft`、normalization最大誤差`5.338e-08`、freeze前truth / episode read
  `0 / 0`で11 technical gatesを全PASS。version 1はraw horizontal ID契約、version 2は
  reduction順序による最大`0.0546875 ft`のparity errorで科学処理前に停止し、version 3では
  exp270と同じfloat64加算順へ戻しただけで閾値や科学条件は変えていない。排他的原因は
  forward transition/prior hysteresisが452 episodes / SSE`59.40%`、backward smoothing
  reversalが86 / `23.04%`、sum-product multiplicityが37 / `9.04%`、state support不足が
  18 / `6.39%`、mixedが45 / `2.12%`、raw-GR / imputation aliasは0。重複条件では
  forwardがSSE`65.78%`、multiplicityが`72.09%`を覆った。predictive truth-odds strong
  wrongはrows`70.35%` / SSE`69.15%`だが、current emissionがtruth oddsを
  `-log(3)`以上悪化させたのはrows`0.253%` / SSE`0.924%`、dominant episode 0。
  filtered rateのzero向きunder-responseはSSE`70.36%`、transition変位誤差とoffsetは
  episode Spearman`0.5693` / SSE加重符号一致`90.22%`。backwardでtruth近傍rate massが
  回復しながらabsolute-position massが悪化する行もSSE`38.33%`あり、rate再同期後も
  累積変位でtranslation gaugeが別datumへlockする機構を直接確認した。したがって過去の
  「GR matchingで違うmodeへ入る」は一部のseed / lock条件として残すが全体root causeから
  降格し、主因をprefix rate prior / sticky transitionの追従遅れ、第二をbackward reversal、
  multiplicityを重複増幅器と確定する。position exact-meanはactual offset方向SSE
  `76.92%`、current量子化biasは`28.24%`で、coarse-grid shrinkageはactualでは誤rate外挿を
  弱めるregularizer側。追加のmessage readoutやmode-ID保持は優先しない。介入する場合は
  rate-change追従を改善しつつstable区間を壊さない単一transition / reset仮説を新規設計し、
  position sigma / exact-mean、GR weight、decoder置換の盲目的gridは行わない。

- 2026-07-25、`exp372_exp287_exp335_feature_union_on_exp264`のKaggle T4 version 2
  （id_no `128530478`）を、1 variant / 3 configs / 5 folds = 15/15 GPU boosters、
  exp264 control・exp287/exp335 standalone・selector再学習・保存feature再生成各0、
  `18425.058989808 sec`で完了した。version 1はprefit loader adapter不足の
  `KeyError: compact_features`でbooster 0停止し、74 unique列をexp264 loader契約へ
  変換する技術修正だけでversion 2を再実行した。technical gateは11/11 PASS。
  `clean273 + saved74 + formation74 + signed23 = 444`のunion CVは
  best standalone exp287 `8.136708220`から`8.071563865`へ`0.065144355 ft`改善し、
  fold条件4/5、formation/signed gainも各5/5 foldsでPASSした。一方、
  `mid_250_1000`はbest standalone比`+0.048399545 ft`、exp264比by-well p95
  `+2.198026177 ft`、worst `fb03ae90 +13.023263266 ft`、clean273比
  `+1/+3/+5 ft`悪化well数`157/53/23`で固定scope/tail gateをFAILした。
  平均的相補性はあるが安全なpromotionではないため、
  decision=`train_complete_guard_failed_closed` / `close_without_same_oof_rescue`。
  完了済みunionをbacklogから削除した。その後ユーザーの明示overrideにより、
  科学FAILを維持したsaved-model CPU inferenceだけを実施した。canonical version 4
  （id_no `128563759`）は`459.376 sec`、14,151 rows / 3 wells、12 candidates、
  parent/signed/union model `40/20/15`、`clean273 + saved74 + formation74 + signed23 = 444`、
  model fit 0で完了した。submission形式はsampleとID内容・順序一致、重複/NaN/Inf 0で
  submit-check PASS。その後ユーザーのscoring完了連絡を受け、Code submission
  `ref=54975325`、submitted `2026-07-25 12:28:12.460000 UTC`、status `COMPLETE`、
  Public LB `7.587`を確認した。exp335 `7.517`比`+0.070`、exp287 `7.530`比`+0.057`、
  exp264 `7.562`比`+0.025`悪化したためML Public-LB anchorはexp335のままとする。
  CV改善がPublic LBへ転移しなかったnegative resultとして、train科学gate FAIL、
  非昇格、同一OOF救済禁止を維持する。外部submitはCodexが実行したものではない。
- 2026-07-25、terminal closedの`exp347_prefix_gr_unary_batched_window_exact_ssm`を再分類せずGPU float32の実用等価性だけを独立監査する`exp393_exp347_practical_numerical_equivalence_audit`を実行した。Stage 0 version 2（id_no `128543320`）は13 gate中10 PASS / 3 FAILで、posterior mean TVT RMSE`0.007435774 > 0.001 ft`、max差`0.191623403 > 0.02 ft`、posterior row-sum error`2.958618e-05 > 1e-05`。decision=`fail_close_without_threshold_dtype_batch_padding_or_kernel_rescue`とexp347 FAILは維持した。その後ユーザーが数値差を受容する明示overrideを行い、fold 0 / seed 42 / neural model 1、LightGBM・booster・PF/Beam・親control再学習0のStage Aをversion 4で完了した。real GR RMSE`22.866144 ft`はshuffle`49.005208 ft`、geometry`32.465005 ft`より良いが、保存済みexp209`12.671087 ft`より`10.195058 ft`悪化。well p95も`43.017463 vs 26.301518 ft`、worst-well regression`75.227871 > 10 ft`で、8/11 checks PASS・3 FAIL。runtime`3.830431 h`、peak`7.495397 GB`、freeze前truth access 0。decision=`close_stage_b_without_exp347_rescue_grid`としてStage B、推論、提出なしでbranchを閉じ、同family rescue gridはbacklogへ戻さない。
- 2026-07-25、`exp367_stratified_signed_curvature_pf`のKaggle private CPU Stage 0 version 1（id_no `128528103`）を`267.914282461 sec`、固定3 signed paths / 5 reporting folds / PF seed-well runs・control replay・model・booster各0で完了した。773 wells中772 wellsの13,631完全512-row blocksを採点し、truth/hidden-role before-freeze 0、SHA readback、identity、PF runs 0を含むtechnical gateは全PASS。overall top1は`0.469591`、MRR gain vs zero-firstは`+0.276771`、selected path RMSE gainはoverall / 1000+ / hidden-like spatial / typewell-purgedで`+0.829601 / +0.911161 / +0.306996 / +0.447147 ft`と正方向だった。一方、circular top1 `0.464016`に対するreal差は`+0.005576 < 0.03`、passing foldsは`2/5 < 4/5`でscientific gateをFAILした。固定pathのGR識別はcircular controlを十分上回らず、98,944 seed-well runsのStage 1 PFへ進む根拠にならない。decision=`stage_0_failed_close_without_rescue`として完了済みexp367をbacklogから削除し、gate緩和、quota/curvature/transition探索、Stage 1、inference、submission、同family救済backlog追加なしでbranchを閉じる。
- 2026-07-25、`exp358_exp209_missing_distance_emission_downweight`はKaggle private CPU Stage 0 version 1（id_no `128528105`）の23/23 technical checks PASS後、別承認されたStage 1を同一kernel version 2で`17475.557881 sec`、fixed missing-distance variant 1 / 5 reporting folds / 773 exact-HMM well-runs、model・booster・PF・Beam・parent control再実行各0で完了した。candidate RMSE `12.012569787`はsaved exp209 `11.938287235`より`0.074282553 ft`悪化し、改善foldは0/5。raw observed/missing `-0.048384/-0.129738 ft`、gap 1--3/4--15/16+ `-0.124171/-0.142545/-0.111850 ft`、1000+ `-0.082776 ft`、hidden-like spatial/typewell-purged `-0.224970/-0.229587 ft`で、仮説対象とrequired scopeが一貫して悪化した。by-wellは358改善/415悪化、p95 delta `+0.469370 ft`、worst `f5859199 +6.630365 ft`、fixed LikPF 50:50も`+0.036981 ft`悪化した。formal technical gateの唯一のfalseはfrozen gzip CSV再読込後のbit-exact weight guardで、753 / 1,200,837 missing rowsの最大差`5.551e-17`、`atol=1e-16`全一致と切り分けた。科学FAILは独立して明確であり、decision=`missing_distance_exp209_failed_close_without_rescue`。完了済みexp358をbacklogから削除し、half-life/floor/hard-mask/sigma/transition/prior/blend rescue、再実行、inference、submission、同family follow-up追加なしでbranchを閉じる。
- 2026-07-25、ユーザー判断によりexp389 Huberをcorrected exp264 fixed12 bankへ`huber_exact_hmm`だけprimary-onlyの13本目として追加する`exp392_exp389_fixed13_dual_selector_on_exp264`を実装し、Kaggle private CPU version 1（id_no `128523057`）を`3666.541645 sec`、1 variant / 2 objectives / outer 5 × inner 4 = 40/40 selector models、parent/control再学習・GPU・downstream TVT・inference・submission各0で完了した。Student-tは併用せずfixed14にはしていない。exp389 6列allowlist、raw/decompressed SHA、global key join、source-fold特徴利用0、truth/error事前読込0、Stage A 153→90 features / compact77、Stage C 25 partitions / 18,919,945 compact rows / 49,191,857 score rows、technical/leakage/score guard、fixed fallback parityは全PASS。Huberは91,035 rows / `2.405795%`、5/5 foldsでtop1利用されたが、fixed13 hard RMSEは親fixed12 `8.652531956→8.769791682`（`+0.117259726 ft`）、改善2/5 folds。1000+ `+0.126035 ft`、hidden-like 2面`+0.160512/+0.154789 ft`、by-well p95`+0.774302 ft`、worst `8902c3f6 +7.875188 ft`で科学gateをFAILした。worst wellのHuber利用率は0%、全well usage-delta Pearsonは`0.004539`で、285 zero-usage wells中158も悪化したため、直接誤選択だけでなく既存候補reranking不安定性の独立根拠となる。H512/whole-well oracleも`0.003663/0.010120 ft`と小さい。decision=`FAIL_CLOSE_FIXED13_SELECTOR_BRANCH`。fixed13 Huber hard候補、same-OOF救済、downstream TVT、current-test生成、inference、submissionなしでbranchを閉じる。
- 2026-07-31、`exp492_huber_exact_hmm_full_replacement_on_exp264`のKaggle private CPU version 1（id_no `129217774`）で、exp264 fixed12のGaussian `exact_hmm` semantic slotをexp389 Huberへ全面置換し、1 variant / 2 objectives / outer 5 × inner 4 = 40/40 selector boosters、parent/control再学習・GPU・downstream TVT・inference・submission各0でStage A/Cと科学readoutを完了した。12 candidate ID/order/domain、4 changed / 8 unchanged parity、固定88/74 schema、exp389 decompressed SHA、global key join、truth-late、technical/leakage/score guardはPASS。hard primaryは`8.652531956→8.639368546`（`-0.013163410 ft`）へ改善したが3/5 foldsに留まり、by-well p95`+0.381470357 ft`、worst `d2f3b1ab +4.254514134 ft`で固定tail gateをFAIL。Huber依存family top1は937,102 rows / `24.764924%`、fixed fallbackは`-0.016115990 ft`のreport-only改善。decision=`FAIL_CLOSE_FIXED12_HUBER_REPLACEMENT_SELECTOR`。科学gate保存後のfeature-importance列名バグでNotebook terminalはERRORとなったがcanonicalコードは修正済みで、結果を変えない追加40-booster rerunは未承認のため行わない。fixed12でもpooled平均改善とwell-tailを両立できず、weight/threshold/domain/gate救済、downstream、inference、submissionなしでbranchを閉じる。
- 2026-07-31、`exp493_student_t_exact_hmm_full_replacement_on_exp264`のKaggle private CPU version 3（id_no `129218034`）で、exp264 fixed12のGaussian `exact_hmm` semantic familyをexp374 df=4 Student-tへ全面置換し、1 variant / 2 objectives / outer 5 × inner 4 = 40/40 selector boosters、parent/control再学習・GPU・downstream TVT・inference・submission各0でStage A/Cと科学readoutを完了した。12 candidate ID/order/domain、4 changed / 8 unchanged parity、固定88/74 schema、global key join、truth-late、technical/leakage/score guardは全PASS。hard primaryは`8.652531956→8.616237400`（`-0.036294555 ft`）、near 0--250 / 1000+ / hidden-like spatial / typewell-purgedもすべて改善したが、改善foldは3/5、by-well p95`+0.540095855 ft`、worst `f6d009f4 +10.472288433 ft`で固定tail gateをFAIL。Student-t依存family top1は1,372,891 rows / `36.281580%`、fixed fallbackは`-0.077883816 ft`のreport-only改善。decision=`FAIL_CLOSE_FIXED12_STUDENT_T_REPLACEMENT_SELECTOR`。v1は親config解決前に0 boosterで停止し、v2は40 boostersと科学gate後のfeature-importance列名バグでERROR、回収不能だったため、承認済みv3を追加40・累計80 CPU boostersで再実行した。pooled平均改善とwell-tailを両立できず、weight/threshold/domain/gate救済、downstream、inference、submissionなしでbranchを閉じる。
- 2026-07-31、`exp496_exp486_absolute_geometry_fixed13_selector_on_exp264`のKaggle private CPU version 1（id_no `129287597`）を`3945.563001 sec`、1 variant / 2 objectives / outer 5 × inner 4 = 40/40 selector boosters、parent/control再学習・PF/HMM/Beam再実行・GPU・downstream TVT・inference・submission各0で完了した。保存exp486 Absolute prediction / mechanism ledger / freeze manifestのpayload・logical・scientific SHA、5 native confidence、global key join、source-fold特徴利用0、truth/error pre-freeze load 0、technical/leakage/score guard、fixed fallback parityは全PASS。fixed13 hard OOFはparent fixed12 `8.652531956→8.461357622`（`-0.191174334 ft`）、4/5 folds、raw observed/missing・high-missing・0--250・1000+・hidden-like 2面の固定7 scopeを全改善し、exp486は420,211 rows / `11.104974%`、5/5 foldsでtop1利用された。一方、by-wellは416改善 / 357悪化、p95`+1.109359862 ft`、worst `14fee784 +9.361781278 ft`で固定tail gateをFAIL。post-freezeではexp486非top1行のincumbent choice change率`34.789662%`、usage-delta Pearson / Spearman`0.020819/-0.000958`、利用0の38 wellsでも24改善 / 14悪化だった。decision=`FAIL_CLOSE_EXP486_ABSOLUTE_FIXED13_SELECTOR`。完了済みexp496をtrain待ちbacklogから削除し、same-OOFのweight/threshold/domain/feature/gate救済、current-test生成、downstream TVT、inference、submissionなしでbranchを閉じる。保存scoreは既存のcross-branch reranking原因診断だけへ別承認で追加可能とする。
- 2026-08-01、`exp501_exp490_mean_reverting_hmm_fixed13_selector_on_exp264`のKaggle private CPU version 2（id_no `129379922`）を`7082.112973 sec`、1 variant / 2 objectives / outer 5 × inner 4 = 40/40 selector boosters、parent/control再学習・HMM/PF/Beam再実行・GPU・downstream TVT・inference・submission各0で完了した。exp490 raw/decompressed SHA、6列allowlist、global key / suffix-offset / exp263 fold repartition、source-fold特徴利用0、truth/error pre-freeze load 0、technical/leakage/score guard、fixed fallback parityは全PASS。fixed13 hard OOFはparent fixed12 `8.652531956→8.264890209`（`-0.387641747 ft`）、5/5 folds、固定7 scopeを全改善し、exp490は2,093,883 rows / `55.335335%`、5/5 foldsでtop1利用された。一方、by-wellは493改善 / 280悪化、p95`+2.904593926 ft`、worst `896d15b9 +18.394664149 ft`で固定tail gateをFAIL。post-freezeではexp490非top1行のincumbent choice change率`35.007153%`、usage-delta Pearson / Spearman`-0.172649/-0.203881`、H512 / whole-well oracle headroom`0.272805/0.355756 ft`だった。decision=`FAIL_CLOSE_EXP490_MEAN_REVERTING_HMM_FIXED13_SELECTOR`。完了済みexp501をbacklogから削除し、same-OOF救済、current-test、downstream TVT、inference、submissionなしでbranchを閉じる。保存scoreは既存P4 cross-fixed13 reranking原因診断だけへ別承認で追加可能とする。
- 2026-07-25、`exp371_exp333_fixed13_dual_selector_on_exp264`はStage Cのwell-level safety FAILを再分類しないまま、ユーザーの「平均で改善しているのなら次に進みましょう」という明示判断により、保存済みfixed13 compactを下流TVTへadd-onlyするStage Dを実行した。変更面は`clean273 + fixed13 compact77 = 350`特徴、1 variant × 3 configs × 5 folds = 15 GPU boosters、保存済みexp264 Stage D v3 control、control再学習0に固定。初回は週次GPU quota、quota回復後はversionless empty shellを解消し、canonical T4 version 1（id_no `128524177`）で15/15 boostersを`13619.488220 sec`で完走した。parent12 compact `8.460811238`に対してfixed13 compactは`8.369996237`、`-0.090815001 ft`、3/5 folds、near / mid / 1000+、hidden-like 2面をすべて改善した。一方、389/773 wells改善、384悪化、by-well p95は`+1.179312073 ft`、worst `e25f1537`は`+4.637599435 ft`で固定`+0.25 ft`上限をFAIL。decision=`STAGE_D_MEAN_IMPROVED_TAIL_GATE_FAILED_CLOSE_NO_INFERENCE`。Stage C FAILを保持し、同一OOF救済、current-test inference、submissionなしでbranchを閉じる。
- 2026-07-25、`exp389_exp209_huber_exact_hmm_emission`のKaggle private CPU version 1（id_no `128466838`）を`19,417.245940 sec`、fixed Huber `delta=1.345` 1 variant / 773 exact-HMM well-runs / reporting 5 folds / model・trained fold・booster・PF・Beam・saved Gaussian control再実行各0で完了した。technical gate、3,783,989 rows、finite、ID、posterior normalization、truth-late join、保存control parityはPASS。directは`11.938287235→11.852741130`へ`+0.085546105 ft`、5/5 folds改善し、raw observed/missing、高missing、1000+、hidden-like 2面、fixed LikPF/HMM 50:50もすべて改善した。一方、362/773 wellsが悪化し、by-well delta p95は`+0.002234 ft`、worst `00bbac68`は`+1.750248 ft`で固定tail gateをFAILした。Student-tのexp374よりtail悪化は小さいが、Huberでも少数wellのwrong-mode固定を完全には防げないnegative resultとして、decision `huber_exp209_failed_close_without_rescue`でterminal closeする。delta/scale/temperature/clip/mixture/Student-t/sigma/missing weight/transition/grid/prior/blend救済、再実行、inference、submission、同family backlog追加なし。低・P4を維持し現行P1/P2を変更しない。
- 2026-07-24、ユーザー確認により、exp357で本来求められていた介入がexp281 residual-offset HMMではなく、exp209 absolute-TVT exact HMMのGaussian row emission単独Huber化だったと確定した。exp357は誤スコープの実行履歴として保持し、その`9.827420 -> 9.737195`を本来の問いへの根拠には使わない。正しい独立実験を`exp389_exp209_huber_exact_hmm_emission`として採番し、親を`exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`、saved Gaussian controlを`11.938287234887435`、変更をfixed Huber `delta=1.345` emissionだけに固定した。absolute TVT、grid/rate/transition/prior、zero-fill population sigma、missing/Type Well GR、momentum、posterior meanはexp209を維持し、0-HMM proxyなし、実行承認後に1 variant / 773 HMM well-runs / model・trained fold・booster・parent rerun各0でactual HMMを評価する契約とした。direct`>=0.05 ft`、4/5 folds、observed/missing/high-missing/1000+/hidden-like/by-well/fixed50:50のAND gateとno-rescueを事前固定した。同日の実装承認によりcompact self-contained train候補、fail-closed inference候補、専用テスト9件を実装し、Huber式、同一emissionでのexp209 exact-kernel parity、truth-late join、SHA/identity、全gate、run未承認guardを検証した。実装時点では正規Notebook採用とKaggle実行を保留し、その後の実行承認・完了・判定は前項のとおりである。inference、submissionは未実施。
- 2026-07-24、`exp381_formation_contact_order_semimarkov_hmm`のKaggle private CPU Stage 0 version 2（id_no `128461656`）を`653.714 sec`、1 diagnostic / 6 reporting surfaces / 5 folds / model・HMM・PF・Beam・booster・parent control再実行・GPU各0で完了した。version 1は`43.463 sec`、ANCC 7 wells / EGFDL 1 wellの列全体欠損を全有限と誤判定してsurface fit前に停止し、formation別finite outer-train donor固定k=10へ修正した。version 2は773 wells、pre-freeze truth / Formation read `0 / 0`、期待15 artifact、SHA manifest raw/decompressed全一致、16-well resourceをPASS。eligibleは349/773、1,291 events、crossing MD MAE/p90 `35.994405 / 61.799226 ft`、順序率`0.997135`、constant surface比`+687.676085 ft`、改善5/5 foldsで各gateをPASSした。一方、contact-TVT RMSEは全foldで上限15 ftを超え、pooled `44.770101 ft`で唯一の必須gate FAIL。decision=`stage0_failed_close_without_semimarkov_hmm`。位置・順序の移送可能性は保持するが、known-prefix単一offsetのcontact-TVT制約は不十分であり、surface k / formation除外 / offset / gate救済、Stage 1 semi-Markov HMM、inference、submissionなしでbranchを閉じる。
- 2026-07-24、`exp357_exp226_huber_emission_independent_audit`はKaggle private CPU canonical version 1（id_no `128448451`）のStage 0を`319.617349 sec`で完了し、pooled gain、fold一貫性、stress非悪化をFAILした。その後、ユーザー明示overrideにより同じkernel version 2でfixed Huber `delta=1.345` actual exact-HMMを1 variant / 773 wells / model・trained fold・booster・親Gaussian再実行各0、`9597.242200 sec`で完了した。saved exp281 Gaussian RMSE `9.827420`をHuber `9.737195`へ`0.090225 ft`改善し、4/5 folds、1000+ `-0.087768 ft`、hidden-like spatial/typewell-purged `-0.208626/-0.001245 ft`を改善した。一方、by-well p95 deltaは`+0.003365 ft`、worst well `4a8ecc0b`は`+1.403715 ft`で固定安全gateをFAILし、exp226 direct ceilingにも`+0.310086 ft`劣った。technical parity、finite、row identityはPASSしているため、Stage 0 proxy FAILでもactual HMMの平均改善は起こり得るが、well-tailとdirect性能が採用水準にないnegative resultと判断する。decision=`stage_1_failed_close_without_rescue`。delta/scale/sigma/tempering/blend救済、再実行、inference、submissionなしでbranchを閉じ、この結果だけを根拠にしたHuber救済backlogは追加しない。
- 2026-07-24、`exp375_exp362_prefix_rate_fixed13_dual_selector_on_exp264`のKaggle private CPU canonical version 1（id_no `128436686`）を`6978.658914 sec`、1 variant / 2 objectives / outer 5 × inner 4 = 40/40 selector models、parent/control再学習・GPU・downstream TVT・inference・submission各0で完了した。exp362 OOF 3,783,989 rows / 773 wellsを6列allowlist、decompressed SHA固定でglobal key joinし、truth/error pre-freeze load 0、missing key 0、source-fold feature利用0、native confidence finite率1.0を確認した。Stage Aは153→90 features、compact 77、Stage Cは25 partitions / 18,919,945 compact rows / 49,191,857 outer-valid score rowsでtechnical / leakage / score guardを全PASS。追加候補top1率は`11.525879%`・5/5 foldsで正、post-freeze oracle headroomはH512`0.162677 ft`、whole-well`0.123213 ft`だった。一方、fixed13 hard OOFはparent fixed12 `8.652531956→8.787855710`（`+0.135323754 ft`）、改善0/5 folds、near/1000+は`+0.037147/+0.148630 ft`、by-well p95`+1.047745 ft`、worst `b19b0395` `+28.995116 ft`でscientific gateをFAILした。hidden-like 2面だけは`-0.080214/-0.071071 ft`改善。worst wellの追加候補top1率は`0.258042%`にすぎず、exp373と同じwellが約`+29 ft`悪化したため、追加候補の直接誤選択だけでなくselector再学習による既存候補reranking不安定性を疑う。decision=`FAIL_CLOSE_FIXED13_SELECTOR_BRANCH`。実装済みexp375をtrain待ち表から削除し、同一OOFのweight/threshold/domain/gate救済、current-test候補生成、downstream TVT、inference、submissionなしでbranchを閉じる。原因確認が必要な場合だけ、exp264/371/373/375の保存scoreを使う0-booster cross-branch reranking診断を低・P4・別承認で検討する。
- 2026-07-24、`exp373_exp355_fixed13_dual_selector_on_exp264`のKaggle private CPU canonical version 1（id_no `128435229`）を`6350.504164 sec`、1 variant / 2 objectives / outer 5 × inner 4 = 40/40 selector boosters、parent/control再学習・GPU・downstream TVT・inference・submission各0で完了した。exp355 Stage 1 OOF 3,783,989 rows / 773 wellsをallowlist 4列だけで読み、raw/decompressed/upstream logical SHA、truth/error pre-freeze load 0、global key join missing 0、source-fold feature利用0を確認した。Stage Aは153→90特徴、compact 77列、Stage Cは40 models / 25 partitions / 18,919,945 compact rows / 49,191,857 outer-valid candidate-score rowsでscore/leakage guardを全PASS。exp355 top1率は`12.319222%`・5/5 foldsで正、hidden-like spatial/typewell-purgedは親比`-0.137525/-0.125127 ft`改善した。一方、fixed13 hard OOFはparent fixed12 `8.652531956→8.695437630`（`+0.042905675 ft`）、改善2/5 folds、near/1000+は`+0.021579/+0.043341 ft`、by-well p95 `+1.008261 ft`、worst `b19b0395` `+29.062587 ft`でscientific gateをFAILした。decision=`FAIL_CLOSE_FIXED13_SELECTOR_BRANCH`。実装済みexp373をtrain待ちbacklogから削除し、同一OOFのweight/threshold/domain/gate救済、downstream TVT、inference、submissionなしでexp355固定13枝を閉じる。独立候補exp375は別仮説として扱う。
- 2026-07-24、`exp371_exp333_fixed13_dual_selector_on_exp264`のKaggle private CPU canonical version 3（id_no `128372803`）を`6761.965850 sec`、1 variant / 2 objectives / outer 5 × inner 4 = 40/40 selector boosters、parent/control再学習・GPU・downstream TVT・inference・submission各0で完了した。exp333 OOF 3,783,989 rows / 773 wellsをglobal key join後にexp263 selector foldへ再partitionし、truth/error pre-freeze load 0、missing key 0、source fold row-count保存、source fold feature利用0、Stage A 153→90特徴、compact 77列、25 partitions / 18,919,945 rows、score/leakage guardを全PASSした。fixed13 hard OOFはparent fixed12 `8.652531956→8.419997371`（`-0.232534584 ft`）、4/5 folds、near / 1000+ / hidden-like 2面を改善し、exp333 top1率は`6.267989%`・5/5 foldsで正だった。一方、fixed fallback `8.238331546`には届かず、400/773 wells改善・373悪化、by-well p95 `+0.861529323 ft`、worst `a48640d9` `+10.757996620 ft`で固定安全gateをFAILした。全wellのexp333使用率とdeltaのPearsonは`-0.070004`で、単純な使用率threshold救済も支持しない。decision=`FAIL_CLOSE_FIXED13_SELECTOR_BRANCH`。実装済みexp371を未着手backlogから削除し、同一OOFのweight/threshold/domain/gate救済、downstream TVT、inference、submissionなしでbranchを閉じる。
- 2026-07-24、ユーザー判断によりexp371のfold policyは`global_key_join_then_exp263_selector_fold_repartition`へ固定した。exp333 saved-exp226 source foldは各well自身に対するOOF provenanceとして保持するがmodel featureには使わず、全3,783,989行を`well_id,row_idx`でglobal key joinして親exp264と同じexp263 selector foldへ再partitionする。fit前に5 selector foldsのmissing key 0、総row parity、source fold row-count保存、source/selector 5×5 overlap、source-fold feature利用0を`exp371_exp333_selector_fold_repartition.json`へ保存する。候補、77 compact、selector設定、1 variant / 2 objectives / outer 5 × inner 4 = 40 CPU boosters、control再学習0は不変。専用8 tests、exp333/exp264/notebookを含む45 tests、py_compile、Ruff、Jupytext、strict validationをPASSし、同canonical kernelのversion 3実行を承認済み。exp333再学習、downstream TVT、GPU、inference、submissionは0。
- 2026-07-24、`exp371_exp333_fixed13_dual_selector_on_exp264`は正規notebook採用と40 CPU booster・control再学習0の承認後、Kaggle private CPU canonical version 1/2を実行したが、どちらもmodel fit前・selector booster 0で停止した。version 1は絶対patternを`Path.glob()`へ渡すpath resolver不備で、修正と回帰test追加後に44 tests / strict validationをPASS。version 2はpath、exp333 file/decompressed SHA、target-free allowlist、3,783,989 rows / 773 wellsを通過した後、exp333 saved-exp226 source foldとexp263 selector foldの不一致をhard guardが検出した。fold別rowsはexp263=`757,738 / 756,650 / 756,255 / 757,101 / 756,245`、exp333=`742,514 / 770,907 / 746,011 / 746,131 / 778,426`で、well `000d7d20`もexp263 fold 0 / exp333 fold 3。全体coverageではなくwell単位fold assignmentが異なる。parent exp264と同じbank semanticsでkey joinしてexp263 foldへ再partitionする、exp263 foldsでexp333 candidate OOFを再学習する、fixed13 routeを閉じる、の選択は結果と計算量に影響するため追加retryを停止し、ユーザー判断待ちとした。version 1/2合計でcontrol再学習、GPU、downstream TVT、inference、submissionはすべて0。
- 2026-07-23、exp333をexp228/exp263の単体置換ではなく固定candidate bankへの追加パスとして再評価する`exp361_exp333_candidate_path_addone_novelty_audit`を、Kaggle private CPU canonical version 2で`234.279773 sec`、1 saved candidate / 5 reporting folds / model・trained fold・booster・親control再実行各0で完了した。exp333 source file/decompressed SHA、exp226 fold parity、exp293 fixed12 candidate bank SHA、H128/H256/H512/whole-well block SHA、truth-before-freeze 0を含むtechnical guardは全PASS。direct `9.076676661`はexp226比`-0.350432936 ft`・5/5 folds改善をparity確認しただけで、exp228 `8.944085501`とexp263 `8.238331715`はhard gateにしなかった。fixed12へのadd-oneはH512 oracle `3.683762664→3.550658788`（`+0.133103876 ft`）、whole-well `+0.102132339 ft`、H512 strict unique-best`11.506357%`、fold改善`5/5`で事前4条件をすべてPASSし、decision=`exp333_candidate_path_novelty_supported`。version 1はCSV round-tripで永続化できない学習時pandas row-hashをhard checkして停止したが、truth/oracle前で科学結果0、version 2では同一file/decompressed SHAとpost-read prediction content SHAへ修正した。exp333はdirect replacementとしては閉鎖を維持する一方、candidate pathとしてcurrent-test生成を行う価値は支持された。次は別承認時だけ新番号を切らずexp333内へ保存済み5 fold modelによる14,151-row candidate inferenceを実装し、prediction artifactまでを検証する。oracle headroomはdeployable selector性能ではないため、単独採用、平均blend、selector変更、submissionへ自動移行しない。
- 2026-07-23、上記exp361の別根拠・ユーザー承認により、新番号を切らず`exp333_exp226_k16_segment_residual_offset_target`内でcurrent-test candidate inferenceを完了した。Kaggle private CPU `kentookumura/exp333-k16-segment-residual-candidate-inference` canonical version 2（id_no `128368525`）は、exp072 v2 stable raw-test replay、exp226 inference v1、保存済みexp333 Stage 1 outer-fold 5 modelを使い、`65.258 sec`で`14,151 rows / 3 wells / 48 K16 segments`を生成した。129 row / 136 model features、train feature schema、model manifest/5 model SHA、saved train summary/OOF SHA、exp226 base、ID/order/finite、raw suffix/K16境界、5-fold meanの全technical guardをPASS。offsetは`-4.249479～+2.592369 ft`、平均`+0.289689 ft`。candidate decompressed SHAは`7571c6281bd2ab484e7bf536a876b8072407b272a0ef0ec5112ca06897a717cd`。version 1はraw replayの205列をexp072 train 196列と誤比較して予測前停止し、version 2はexp072正規allowlist適用だけで修正した。新規model/booster、parent/control再学習、selector/blend/fixed12平均、`submission.csv`、competition submitは0。これにより`exp333_current_test_segment_offset_candidate_generation`は完了済みとしてbacklogから削除する。次の組み込み方法はユーザー選択により13候補selector再学習へ固定し、`exp371_exp333_fixed13_dual_selector_on_exp264`としてsteering、fixed13 contract、Stage A + Stage C Jupytext候補、専用testsまで実装した。target-free safety gate案は同時に試さない。exp371は1 variant / 2 objectives / outer 5 × inner 4 = 40 CPU boosters、parent/control再学習0、downstream TVT/GPU/inference/submission各0の実行scopeで、正規notebook採用とKaggle run承認待ち。
- 2026-07-23、exp352のsame-group平均signal`+0.381540 GR API`を直接補正へ採用せずsoft quality featureとして独立検証するため、`exp353_typewell_group_quality_feature_preflight`の0-booster Stage 0をKaggle private CPU version 1（id_no `128362932`）で完了した。primary 1 + stable group-label shuffle 1 / 5 reporting folds / LightGBM config・trained fold・booster・親exp148 control再学習各0、runtime`112.107959 sec`。exp065 membershipとexp148 summary/by-well SHAをhard preflightし、3,783,989 score rows / 773 wells、feature freeze前outer-valid truth 0、全6 feature finite、coverage`0.980595`、fallback`0.019405`、real相関の正方向4/5 foldsはPASSした。一方、residual sigma対exp148 well-RMSEのpooled Spearmanは`0.006134 < 0.15`、q4-q1は`+0.202701 < +0.25 ft`、shuffle Spearman`0.065301`に対するreal-minus-shuffleは`-0.059166 < +0.05`で、8 checks中5 PASS・総合FAIL。native Type Well群のsupport/noise/reliabilityはexp148のwell errorをgroup固有に安定識別しないnegative resultとして、列選択、group/fallback/threshold救済、Stage 1の15 GPU boosters、再実行、raw-test再生成、inference、submissionなしでbranchを閉じ、backlogから削除する。exp352のdirect prior不採用と旧exp314閉鎖を維持し、同familyの救済候補は追加しない。
- 2026-07-23、`exp360_typewell_reference_shift_zncc_confidence_readout`の0-booster Stage 0をKaggle private CPU canonical version 2（id_no `128366385`）で`125.393474 sec`、real ZNCC 1 / stable permutation 1 / saved raw Gaussian 1 / 5 reporting folds / model・trained fold・booster・PF/Beam/HMM・親control再実行各0で完了した。version 1はcompetition dataのcanonical `/kaggle/input/competitions/.../train`候補不足によりscore生成前に停止し、version 2では入力root候補だけを直した。3,783,989 rows / 773 wells / 7,787 blocks、truth-before-freeze 0、coverage`0.988828`、fold Q1/Q4非重複、SHA manifest 13件一致は確認したが、`896d15b9`にsupported blockがなく772/773 wellsでtechnical FAIL。primary `best_nonzero_minus_zero_zncc`はQ4-Q1 mean/median block RMSE差`+0.107479 / +0.085354 ft`、正方向4/5 folds、pooled bad10 AUC`0.505164`、1000+`-0.169027 ft`で、raw Gaussian AUC`0.549949`比`-0.044785`・勝利1/5 folds、permutation AUC`0.488520`比も`+0.016644 < +0.02`となりscientific FAIL。scale/offset不変なshape scoreへ置換してもexp264 bad blockの安定confidenceにならないnegative resultとして、事前規則どおりthreshold/family/shift grid/pair/std/sentinel/supporting family救済、add-only特徴化、prediction変更、再実行、inference、submissionなしでbranchを閉じ、backlogから削除する。exp340の閉鎖も維持し、この結果だけを根拠とする同family救済候補は追加しない。
- 2026-07-23--24、`exp355_exp226_dip_rate_prior_on_exp209`の0-HMM Stage 0をKaggle private CPU canonical version 1（id_no `128366148`）で`460.872765 sec`、1 diagnostic / 5 reporting folds / HMM・model・trained fold・booster・親control再実行各0で完了した。3,783,989 rows / 773 wells / K16 12,368 segments、fallback 0、入力・freeze・成果物SHAは全一致。segment rate-change RMSEは`0.018237982→0.016710597`（`+8.374744%`、4/5 folds）、cumulative path RMSEは`49.493155→46.977325 ft`（`+2.515830 ft`、5/5 folds）だったが、worst `071d7b45`が`+69.017669 ft`で総合FAIL。その後ユーザーが平均改善を根拠にworst-well gateを明示overrideし、同じcanonical version 2で1 candidate / 773 exact-HMM well-runs / 5 reporting folds / model・booster・親control再実行各0を`18,161.789478 sec`で完了した。technical gateは全PASS。directはexp209 `11.938287235→11.291976616`（`+0.646310619 ft`、`+5.4138%`、5/5 folds）、fixed LikPF 50:50も`10.269696317→10.053143746`（`+0.216552571 ft`、4/5 folds）とpooledでは改善した。一方、360/773 wells改善に対して413 wellsが悪化し、hidden-like spatial / typewell-purgedは`+0.414943459 / +0.371719953 ft`、worst `86454a6f`は`+52.743754462 ft`で、8 scientific checks中5 PASS・総合FAIL。Stage 0と異なるwellで大幅悪化し、平均signalは実在してもhidden-test-like transferとwell-level safetyがないnegative promotion resultとして、parameter/clip/blend/selector救済、再実行、inference、submissionなしでbranchを閉じ、backlogから削除する。exp355固有の重複救済案は追加しない。
- 2026-07-24、`exp362_segment_local_donor_slope_exact_hmm`のKaggle private CPU version 1（id_no `128368310`）を`19,777.653141 sec`、1 scientific variant / 5 reporting folds / 773 HMM well-runs / model・trained fold・booster・GPU・親control再実行各0で完了した。notebook technical gateは3,783,989 rows / 773 wells、finite、fold donor exclusion、truth-before-freeze 0、exp226 resolve 0、parent SHA、posterior正規化、runtimeをPASS。pooled RMSEはexp209 `11.938287235→11.161677223 ft`（`+0.776610012 ft`）、1000+ `+0.858876684 ft`、hidden-like spatial/typewell-purged `+0.273186542/+0.351965543 ft`と改善したが、fold 1/4が悪化して3/5 folds、worst `86454a6f`が`+52.741425793 ft`で科学gateをFAILした。さらにtarget prior実ファイル監査でlocal gradient採用は`0/12,368`、全`mu_rate`がprefix rateと一致し、reasonはeffective donors不足11,596 / nearest distance超過772だった。保存fallback列0はprefix側同名fieldによる上書きバグで無効。このscoreはlocal donor-slope介入ではなくprefix-rate-only residual HMMの参考値であり、post-run support audit FAILとしてbranchを閉じる。support/bandwidth/K/ridge/fallback/HMM/blend救済、記録バグ修正だけの再実行、inference、submissionは行わない。完了済みexp362をtrain待ち表から削除し、同じK16 donor supportに依存するexp356は非退化supportの独立証拠までblocked/demotedとする。
- 2026-07-23、ユーザー指定の5件を旧closed実験のreopen / reparentではなく、相互依存しない新番号のdesign-only後継として固定した。`exp355_exp226_dip_rate_prior_on_exp209`（旧exp323）はfailed exp307--309/338 chainを外し、exp209のconstant rate-prior meanだけをfold-safe exp226 K16 geometry rate scheduleへ置換する0-HMM identifiability readout、`exp356_exp226_donor_covariance_sig_r_on_exp209`（旧exp324）はexp355に依存せずexp209の`sig_r=0.002`だけをK16 donor covariance scheduleへ置換する0-HMM NLL readout、`exp357_exp226_huber_emission_independent_audit`（旧exp344）はexp342 activationを外してexp281/保存exp280 Gaussianへfixed `delta=1.345` Huberだけを加える0-HMM shift-rank readout、`exp358_exp209_missing_distance_emission_downweight`（旧exp308）はexp307 finite-MADを外してexp209のraw-missing row log emissionだけを`max(0.25,2^(-distance/8))`で弱める0-HMM technical audit、`exp359_exp226_window_likelihood_on_exp281`（旧exp325）はexp323/338 chainを外してexp281へ500-row/stride 125のfixed sparse window potentialだけを加える0-HMM rank readoutとする。exp281系のexp357/359は親がpromotion FAIL済みのため、conditional Stage 1にexp281比`0.05 ft`改善だけでなくexp226 direct RMSE `9.427109596582213`以下のabsolute ceilingを追加した。全5件ともStage 0はHMM/model/trained fold/booster各0、Stage 1は全gate PASSと別承認時だけ1 variant / 773 HMM runs、parent control再実行0。追加依頼でexp355はStage 1まで実行したがhidden-like 2面とworst-well guardをFAILした。exp357もStage 0 FAIL後の明示overrideでStage 1 actual HMMまで実行し、exp281比`0.090225 ft`・4/5 folds・required scopeを改善したが、by-well p95、worst-well、exp226 direct ceilingをFAILした。両方とも救済、inference、submissionなしで閉鎖した。2026-07-25にexp358/359はStage 0 compact self-contained候補とfail-closed inference候補まで実装した。exp358はStage 0の23/23 technical checks PASS後、別承認されたStage 1 actual HMMを実行したが、exp209比`-0.074283 ft`、0/5 folds、required scopeとtail/fixed-blend guardを全てFAILし、rescueなしで閉鎖した。exp359も同日に正規Notebookを採用してKaggle CPU Stage 0を完了したが、saved exp280 control比MRR`-0.022264`、top3`-0.033496`、改善fold各0/5、stress 3面負方向で固定gateをFAILし、Stage 1・inference・submissionなしで閉鎖した。exp356はexp362 support監査を受け非退化support証拠までblocked/demoted、各優先度は変更しない。
- 2026-07-23、旧Type Well群branchの内容を一律に無効とはせず、旧実験をreopen / reparentしない独立後継として3件を切り出した。`exp352_typewell_transfer_safety_guard_readout`は保存済みexp311群統計に固定availability/support/fallbackだけを適用する0-model安全性readoutで、Kaggle CPU version 1（id_no `128360039`）を1 diagnostic / 3 surfaces / 5 reporting folds / model・booster・decoder・HMM各0、親control再実行0で完了した。exp311 summary/pair SHA hard preflight、manifest freeze前truth 0、identity parity 0、exact coverage`0.972833`、same-group gain`+0.381540 GR API`、5/5 folds、leave-group-out / spatial+typewell-purged negative transfer`-0.164862 / -0.496752 GR API`はPASS。一方、exact groupを許可した`d07aed8f`がidentity RMSE`5.587119`からguarded`18.501835`へ悪化し、worst`+12.914716 GR API`で上限`+0.25`を大幅超過したため総合FAIL。support availabilityだけでは個別well safetyを保証できないnegative resultとして、threshold/fallback/global重み救済、再実行、inference、submissionなしでbranchを閉じ、exp311/312と旧exp314--320の閉鎖を維持する。`exp353_typewell_group_quality_feature_preflight`はdirect priorから独立したsoft quality featureの0-booster preflightとして後続実行し、上記のfixed shuffle gateをFAILして閉鎖した。`exp354_typewell_group_candidate_family_prior_readout`はexp311/312/313/315出力を禁止する独立P2として、Kaggle private CPU version 1（id_no `128363177`）をreal prior 1 + stable group-label shuffle 1 / 5 reporting folds / model・booster・candidate再生成・親control再実行各0で完了した。exp293 v2固定12 candidateを6 familyへ集約し、target-free freeze前truth 0、fit-valid overlap 0、coverage`0.980595`、real family rank Spearman`0.325789`、5/5 folds、hidden-like spatial/typewell-purged`0.381736 / 0.376570`はPASSしたが、shuffleも`0.327079`となりreal-minus-shuffleは`-0.001290 < +0.05`でFAILした。native Type Well group固有signalではなくglobal family base rateが主成分のnegative resultとして、family/support/group/rank metric救済、Stage 1 40 selector models、再実行、inference、submissionなしでbranchを閉じ、実装済みtrain待ち表から削除する。exp353/354はいずれもgroup固有signalのshuffle差を満たさず閉鎖し、自動昇格しない。旧exp320の相関evidence着眼点は`exp343_acf_effective_sample_likelihood_tempering_audit`をType Well群非依存の独立後継として扱う。
- 2026-07-23、P1--P2の`exp340_exp226_depth_alias_block_confidence_readout_on_exp264`をKaggle private CPU version 1（id_no `128356047`）で`26.400168 sec`、7 fixed family / circular control 1 / model・trained fold・booster・HMM・親control再実行各0で完了した。3,783,989 rows / 7,787 blocks / 773 wellsのSHA、truth-free freeze、coverage、Q1/Q4非重複を満たしtechnical gateはPASS。zero rank、absolute top1 shift、prior-block jump、3-block sign inconsistencyはQ4-Q1 mean block RMSE差`+0.905341 / +1.359545 / +2.253795 / +1.513854 ft`、各5/5 folds正方向だったが、必須bad10 AUCは`0.541894 / 0.544737 / 0.574392 / 0.548155`で全て`0.60`未満。最良prior-block jumpもcircular control勝利3/5 foldsで、全7 familyが固定AND gateをFAILした。shift形状は平均誤差層を分けても安定した10 ft failure detectorにはならないnegative resultとして、threshold/family blend/補正/selector/inference/submission/再実行なしでbranchを閉じ、backlogから削除する。同family救済は追加せず、独立P3のexp342とexp343を後置した。その後exp342/343とexp354はStage 0 FAILで閉鎖した。
- 2026-07-23、`exp335_signed_residual_meta_on_exp264`のKaggle T4 Stage D canonical version 2（id_no `128232946`）を`20,017.035909 sec`、1 variant / 3 LightGBM configs / 5 folds = 15/15 GPU boosters、saved control再学習0で完了した。clean273 + saved74 + signed23の370特徴はsaved exp264 `8.460811→8.146108`（`0.314703 ft`改善）、4/5 folds、0--250 / 250--1000 / 1000+ / hidden-like 2面をすべて改善し、signed23も非ゼロgain・最大feature share`25.296%`だった。一方、345/773 wellsが悪化し、by-well delta p95 `+1.728657 ft`、worst `fb03ae90` `+10.238752 ft`で固定scientific guardをFAILした。clean273比worst deltaも`+14.482873→+17.774910 ft`、`+1/+3/+5 ft`悪化well数も`135/39/14→150/53/21`へ増え、promotion guardもFAIL。平均signalは有効だがtailへ安全に一般化しないnegative promotion resultとして、gate緩和とsigned objective/grid/threshold/特徴救済を行わず、完了済みbacklogを削除する。同familyの救済候補は追加しない。その後のユーザー明示overrideでは、保存済み40/20/15 modelsを使うKaggle CPU inference canonical version 3（id_no `128358534`）を`387.808 sec`、学習booster 0、internet/GPU offで完了した。14,151 rows / 3 wells、final 370特徴、formula/top-1 parity `0.0`、全model SHAを確認し、sampleとID順まで一致する`submission.csv`がWARN/FAILなしでsubmit-checkをPASSした。ユーザー実施のcode submission ref `54928806`はPublic LB `7.517`でCOMPLETEとなり、exp287 `7.530`を`0.013`、exp264 `7.562`を`0.045`改善して追跡中のPublic-LB reference anchorを更新した。ただしtrain-side非promote判断は変更せず、既存候補の優先度も変更しない。
- 2026-07-23、`exp346_exp209_observed_only_finite_sigma_gr_hmm`のKaggle private CPU version 1（id_no `128227279`）を`17,757.849174 sec`、1 variant / 773 HMM well-runs / control再実行・model・booster各0で完了した。technical gateは3,783,989 rows、finite 100%、ID mismatch 0、fallback 0%、raw missing emission parity差0、baseline metric parity、posterior正規化、runtimeを全PASS。一方、directはexp209 `11.938287→13.295027`（改善`-1.356739 ft`）、1/5 folds、raw observed/missing `-1.647067 / -0.710366 ft`、high-missing、1000+、hidden-like 2面も全悪化した。fixed LikPF 50:50も`10.269693→10.531118`（`+0.261425 ft`悪化）、p95差`+1.162673 ft`、worst `be83e781`は`+70.766418 ft`。finite sigma中央値はexp209幅の`0.369701`倍まで縮み、observed行の過信が全系列posteriorを通じてmissing行にも波及したと解釈する。decisionは`observed_only_finite_sigma_failed_close_without_rescue`。sigma/confidence/emission/HMM/blend救済、inference、submissionなしでbranchを閉じ、完了済みbacklogを削除する。同familyの新規救済は追加せず、独立0-boosterのexp340をP1--P2、ACF安定性を先に監査する既存exp343を低-中P3のまま維持する。
- 2026-07-22、`exp345_exp209_time_varying_gr_affine_calibration_hmm`のKaggle CPU Stage 0 canonical version 2を`4871.012861秒`、parent 773 + affine variant 773 = 1,546/1,546 HMM runs、494,720 rows / 773 wellsで完了した。technical gateは全finite、fallback 0%、posterior正規化誤差`3.22e-15`、runtime 1.3531時間でPASS。candidateはparent `14.501048`から`14.331543`へ`+0.169505 ft`改善し、4/5 folds、GR predictive NLL `4.651670→4.646152`、boundary jump p95 `0.010089 sigma`もPASSした。一方、hidden-like spatial / typewell-purgedの必須readoutが生成されず、373/773 wells改善に対して400/773 wellsが悪化、worst well `c03b9305`は`+9.354827 ft`で上限`+0.25 ft`を大幅超過したためscientific AND gateをFAILした。promotion gate SHAは`39296d1b...40525a`、decisionは`stage_failed_close_without_rescue`。Stage 1、version 3、affine/process-noise/grid救済、inference、submissionなしでbranchを閉じ、完了済みbacklogを削除する。exp338とは独立兄弟のままで、相互の判断やsuccessor chainへ影響させない。同familyの救済backlogは追加せず、再訪には独立根拠、別実験の事前設計、ユーザー確認を要求する。
- 2026-07-23、`exp350_exp345_bidirectional_gr_affine_smoother`のKaggle private CPU Stage 0 canonical version 1（id_no `128274195`）を`2749.356610 sec`、1 variant / forward 773 / bidirectional RTS smoother 773 / candidate HMM 773、control HMM再実行・LightGBM・booster・PF・Beam・GPU各0で完了した。technical gateはexp345保存成果物SHA、parent/causal metric差0、forward schedule parity最大差`7.070433e-11`、494,720 rows / 773 wells、全finite、terminal identity、covariance PSD/contraction、scale clip 0%、runtimeを全PASS。candidateはmasked parent `14.501048→14.367548`（`+0.133499 ft`）、5/5 folds、hidden-like 2面改善だったが、exp345 causal `14.331543→14.367548`（`-0.036006 ft`）、2/5 foldsでFAILした。parent比403/773 wells改善、370悪化、median delta`-0.008672 ft`に対しp95`+1.346427 ft`、worst `8995c945`は`+20.887374 ft`で、tail抑制の目的に反した。数値異常ではなくfuture GRによる平滑化が一部wellの誤calibrationをprefix方向へ逆伝播したscientific negativeと判断し、decisionは`stage_0_failed_close_without_rescue`。Stage 1、version 2、Q/rcond/clip/grid、causal blend、row/well gate、inference、submissionなしでbranchを閉じ、完了済みbacklogを削除する。同familyの救済候補は追加せず、独立0-boosterのexp340をP1--P2として維持する。
- 2026-07-22、GR尤度まわりの未解決点を原因別に分離し、`exp339_missing_gap_pseudomask_uncertainty_readout`、`exp340_exp226_depth_alias_block_confidence_readout_on_exp264`、`exp341_missing_gap_calibrated_soft_variance_exp226_residual_hmm`、`exp342_exp226_student_t_residual_offset_emission_audit`、`exp343_acf_effective_sample_likelihood_tempering_audit`、`exp344_exp226_huber_residual_offset_emission_audit`の6件をdesign-onlyで作成した。優先順は、0-HMMで補間誤差を直接測るexp339、0-boosterでdepth alias検知可能性を測るexp340、exp339全gate通過時だけのsoft variance HMM exp341、固定df=4 Student-tのshift-rank先行監査exp342、known-prefix ACF安定性を先に測るexp343、exp342が「極端残差改善・全体flattening失敗」の事前指定patternになった場合だけの固定delta=1.345 Huber exp344とする。exp341/342/343/344のfull 773-well HMMはそれぞれStage 0全gate通過と別承認を必須とし、既存controlは再実行しない。6件ともGroupKFold、truth-free freeze、long-tail guard、単一変更、fail-closed依存をsteering/configへ固定した。その後、exp339はStage 0固定gate FAILでexp341とともに閉鎖、exp340もStage 0の7/7 family FAILで閉鎖した。exp342もStage 0固定gate FAILで閉鎖し、flattening依存が不成立のexp344も未実行で閉じた。exp343はStage 0実装済み・未実行として残す。
- 2026-07-22、`exp339_missing_gap_pseudomask_uncertainty_readout`のKaggle private CPU Stage 0 version 1（id_no `128226213`）を`320.79614 sec`、scientific readout 1 + controls 2 / 5 outer folds / HMM・model config・trained fold・booster・親control再実行各0で完了した。116,458 pseudo-gap rows / 773 wells、coverage全fold1.0。2D tableのprimary NLL `4.041311`はglobal constant `4.075874`よりpooled・5/5 foldsで良く、variance/MSE比`0.975582`とfold別校正5/5、length-sigma Spearman `0.518407`と正相関5/5もPASSした。pooledではcircular `4.044584`より良かったが、fold別real placement勝利は0・1の2/5だけで固定4/5 gateをFAILした。table content SHA `6a9bd955...b4e5`は再現性証拠としてのみ保持し、自然欠損への転送表として採用しない。bin/support/pseudo-gap数/補間法の救済、再実行、HMM、inference、submissionなしでexp339を閉じ、依存するexp341も閉じる。同系救済backlogは追加せず、次は独立0-boosterのexp340をP1--P2として維持し、exp342/343は低-中P3のまま後置する。
- 2026-07-23、ユーザー依頼により`exp342_exp226_student_t_residual_offset_emission_audit`のStage 0だけを実装した。SHA固定済みexp280 Gaussian target-free scoreを再生成なしのcontrolとし、同じ512-row/13-shift/fold/missing/exp281 sigmaで固定`df=4` Student-t scoreを生成するcompact self-contained train、fail-closed inference、正規Notebook、専用contract testを作成した。Student-t/control bundleをtruth join前にfreezeし、同一stable nonzero circular rotation、1000+、hidden-like 2面、persistent-offset、truth-nearest shiftに`|z|>=3`が1行以上あるblockのtop3/regretをAND gateへ固定した。Stage 0はscientific score 1 / saved control 1 / HMM・model config・trained fold・booster・control再生成各0。py_compile、Ruff、pytest `7 passed`、Jupytext test、strict exp validationをPASSした。Kaggle package/push/run、Stage 1、inference、submissionは未承認・未実施で、P3を維持する。
- 2026-07-23、ユーザー依頼により`exp343_acf_effective_sample_likelihood_tempering_audit`のStage 0だけを実装した。SHA固定済みexp226 OOFからgroup-safe `well_id/fold`だけを読み、raw finite known-prefix GR residualのcontiguous run内pairwise Pearson ACFをlag 1--20で計算するcompact self-contained train、fail-closed inference、正規Notebook、専用contract testを作成した。last-512はknown-prefix末尾512 raw rowsを先に固定し、finite residual 128未満・各lag pair 20未満・rho非finiteをouter-train fold median fallbackとする。`tau_raw=1+2 sum(max(rho,0))`、support 200 log shrink、clip`[1,4]`を固定し、full/last-512 stabilityは両window raw-evaluable wellだけで計算する。window別median tau、upper clip率、fold median比は悪い側をAND gateに使う。Stage 0はdiagnostic 1 / 5 reporting folds / HMM・model config・trained fold・booster・control再生成各0。py_compile、Ruff、pytest `7 passed`、Jupytext test、strict exp validationをPASSした。実装時点ではKaggle package/push/run、Stage 0科学値、Stage 1、inference、submissionは未承認・未実施で、P3を維持した。
- 2026-07-23、その後`exp343_acf_effective_sample_likelihood_tempering_audit`のKaggle private CPU Stage 0 version 1（id_no `128358348`）を`273.667045 sec`、diagnostic 1 / 5 reporting folds / HMM・model config・trained fold・booster・親control再実行各0で完了した。773 wells中joint-evaluableは295（`0.381630`）で下限0.90をFAIL、fallbackは478（`0.618370`）で上限0.10をFAIL、stable foldは0/5だった。raw tauのouter-train fold中央値はfull `9.771436--10.039963`、tail `24.258286--25.172847`で、`tau_eff`はfull `99.7413%`、tail `100%`が上限4へclipされ、full/tail Spearmanは定数列のためundefinedとなった。log ratio 0.0やfold median比1.0はclip由来で安定性の証拠に使わない。decisionは`stage_0_failed_close_without_rescue`。一律`tau_eff=4`の採用、lag/support/clip/temperature/downsampling救済、Stage 1、inference、submission、再実行なしでbranchを閉じ、実装済み・Kaggle train待ち表から削除する。同family successorは追加しない。その後exp353/354もStage 0 FAILで閉鎖し、既存`exp277`を自動昇格させない。
- 2026-07-23、`exp342_exp226_student_t_residual_offset_emission_audit`のKaggle private CPU Stage 0 version 1（id_no `128356155`）を`468.127417 sec`、scientific score 1 / saved Gaussian control 1 / 5 reporting folds / HMM・model config・trained fold・booster・control再生成各0で完了した。773 wells / 3,783,989 rows / 7,787 blocks、coverage、saved Gaussian parity、circular gapはPASS。Student-tはGaussian比でMRR `+0.000666`、top3 `+0.001156`、fold改善5/5・4/5だったが、必須pooled gain各`+0.01`に届かず、1000+・hidden-like 2面・persistent-offsetを束ねたstress非劣化もMRR/top3両方FAILした。`|z|>=3`の174 blocksではtop3 `+0.022989`、mean regret `-0.692701 ft`と改善したが、全体採用条件を満たさない。flattening signalはfalseで、事前指定したexp344 dependency patternも不成立。Stage 1、df/scale/temperature/Huber等の救済、再実行、inference、submissionなしでexp342を閉じ、exp344も未実装・未実行で閉じる。独立したexp343は後続のStage 0固定gate FAILで閉鎖済みである。
- 2026-07-24、その後ユーザーの明示overrideにより、`exp342_exp226_student_t_residual_offset_emission_audit`のfull HMM Stage 1を同じcanonical Kaggle private CPU kernel version 2（id_no `128356155`）で`14,789.392992 sec`、固定`df=4` Student-t 1 variant / 773 HMM well-runs / model・trained fold・booster・Gaussian control再実行各0で完了した。変更はexp281のGaussian行別emissionからStudent-tへの置換だけで、offset/rate grid、transition、prior、sigma、missing、posterior meanを固定した。Student-tは保存済みexp281 GaussianをRMSE `9.827420→9.779772`へ`+0.047648 ft`改善したが、必要`+0.05 ft`に`0.002352 ft`届かず、改善foldも3/5だった。1000+は`+0.049643 ft`改善した一方、hidden-like spatial/typewell-purgedは`0.014174 / 0.220136 ft`悪化し、by-well delta p95は`+1.063793 ft`、worst `77b0d905`は`+12.893602 ft`、exp226比も`+0.352662 ft`悪化した。technical parity、finite、row identityはPASSしているため、Stage 0 proxy FAILでも実HMMが小幅改善し得ることは確認したが、full HMM自身の改善量・一貫性・tail safetyが不足するnegative resultである。decisionは`stage_1_failed_close_without_rescue`。df/scale/temperature/grid/Huber/cap/missing/ACF/blend救済、再実行、inference、submissionなしでterminal closeし、exp344閉鎖を維持する。独立後継exp357の設計判断は変更せず、exp342固有の救済backlogは追加しない。
- 2026-07-24、ユーザーが意図していた比較を「exp209 absolute-TV​T exact HMMのGaussian emissionだけを固定`df=4` Student-tへ置換」と再確定し、exp342のexp281 residual-offset結果とは独立した`exp374_exp209_student_t_exact_hmm_emission`を実装・実行した。Kaggle private CPU version 1（id_no `128436182`）を`19,662.082424 sec`、1 variant / 773 HMM well-runs / model・trained fold・booster・Gaussian control再実行各0で完了。technical gate、3,783,989 rows、finite、ID、posterior normalization、truth-late join、保存control parityはPASSした。directは`11.938287235→11.720478702`へ`+0.217808533 ft`、4/5 folds改善し、raw observed/missing、高missing、1000+、hidden-like 2面、fixed LikPF/HMM 50:50もすべて改善した。一方、343/773 wellsが悪化し、by-well delta p95は`+0.982661 ft`、worst `a6f967fb`は`+35.015963 ft`で固定tail gateを大幅にFAILした。平均的にはStudent-tが有効でも少数wellのwrong-mode固定を安全に抑えられないnegative resultとして、decision `student_t_exp209_failed_close_without_rescue`でterminal closeする。df/scale/temperature/clip/mixture/Huber/sigma/transition/grid/blend救済、再実行、inference、submission、同family backlog追加なし。低・P4を維持し現行P1/P2を変更しない。
- 2026-07-24、その後ユーザー判断でexp374の平均改善を単独昇格ではなくselector候補として検証する`exp388_exp374_fixed13_dual_selector_on_exp264`を実装し、corrected exp264 fixed12 bankへ`student_t_exact_hmm`をprimary-onlyの13本目として追加した。Kaggle private CPU version 1（id_no `128464582`）を1 variant / 2 objectives / outer 5 × inner 4 = 40 CPU models、parent/control再学習・GPU・downstream TVT・inference・submission各0、`7,253.168438 sec`で完了。technical/leakage/score guardはPASSし、expected-error MAE、within10 logloss/Brierはprior比pooled・5/5 folds改善した。Student-tは692,647 rows、pooled`18.304678%`、5/5 foldsでtop1利用され、H512/whole-well oracleにも`0.097299 / 0.073408 ft`の補完性があった。一方、fixed13 hard RMSEは親fixed12 `8.652531956`から`8.736104109`へ`+0.083572154 ft`悪化し、改善2/5 folds。1000+ `+0.089724 ft`、hidden-like 2面`+0.088184/+0.091252 ft`、by-well p95`+0.910123 ft`、worst `d2f3b1ab +6.708956 ft`をFAILした。候補の局所補完性を現行hard selectorが安全なgainへ変換できないnegative resultとして`FAIL_CLOSE_FIXED13_SELECTOR_BRANCH`で閉じ、same-OOF weight/threshold/domain/gate救済、downstream TVT、inference、submissionを行わない。exp371/373/375/388のfixed13失敗を踏まえ、再訪はStudent-t TVTのhard選択ではなくGaussian--Student-t disagreement/std/loglikのcontinuous add-only risk featureに限定し、低・P4とする。
- 2026-07-22 に `finite_only_robust_sigma_gr` / `exp307_finite_only_robust_sigma_gr` のKaggle private CPU version 2（id_no `128085112`）を27,402.239秒、2 variants / 1,546 HMM well-runs / control再実行・model・booster各0で完了した。version 1は1,546 runs後のsaved LikPF列契約ミスで停止し、version 2は`last_known_tvt + likpf_mean_d`復元とHMM前schema guardを追加して完走した。finite std directは`11.938287 → 14.209718`（改善`-2.271430 ft`、0/5 folds）、finite MAD primaryは`11.938287 → 15.661341`（改善`-3.723054 ft`、0/5 folds）。fixed LikPF 50:50もそれぞれ`+0.497797 / +0.917640 ft`悪化した。旧0-fill scale中央値`38.6418`に対しfinite std `13.8957`、finite MAD `10.1367`まで縮み、MADは365/773 wellsで下限10に張り付いたため、GR emissionの過信が主因と解釈する。1000+、hidden-like 2面、p95、worstも全FAIL。saved LikPF baseline parityの約`3e-6 ft`不一致は候補悪化より十分小さくnegative decisionは不変である。sigma/clip/likelihood/HMM/blend救済、inference、submissionなしでexp307を閉じる。exp307 PASSを必須としたexp308/310、exp308 PASS依存のexp309、同じ固定lineageのexp323--328も未実行のまま閉鎖し、該当項目をbacklogから削除する。将来別parentへ再設計する場合は独立根拠、事前設計、ユーザー確認を必要とする。
- 2026-07-22、ユーザー依頼により`prefix_backtested_structure_sigma_gr` / `exp337_prefix_backtested_structure_sigma_gr`をdesign-onlyで作成した。Gaussian GR emissionを持つexp209を科学的親、exp307を失敗根拠とし、finite observation varianceへknown-prefix時系列後半のzero-center予測MSEから得る`tau_structure^2`を加え、`sigma_eff^2=sigma_finite^2+tau_structure^2`とする単一変更を固定した。内部splitは60/40、rolling originは60%/80%、各forward blockは20%、finite pair合計50未満またはearly/late各20未満は同prefixのexp209 zero-fill scaleへno-op fallback、clipは`[10,60]`。Stage 0は1 diagnostic / HMM・model・booster各0でfinite-onlyとzero-fillに対するforward Gaussian NLLを監査し、全gate PASSと別承認時だけStage 1の1 variant / 773 HMM well-runs / control再実行0を許可する。exp305/307のstrong negativeを踏まえ優先度は低-中P2とし、現行P1 downstream候補より後、P3 failure readoutより前。実装、Notebook編集、Kaggle package/push/run、inference、submissionは行っていない。
- 2026-07-23、`exp338_exp209_well_adaptive_transition_noise`のKaggle private CPU version 3（id_no `128226900`）を1 variant / 773 HMM well-runs / 3,783,989 rows / model・booster・PF・Beam・control再実行各0、`11,376.512秒`で完了した。v1は親raw/local metrics schema差でHMM前ERROR、v2は773/773 HMM後にexp115の正式な`purged_train_excluded`をlate role契約が拒否してERRORとなり、科学式・gateを変えず契約だけ修正した。v3のbaseline parity、finite、ID、posterior normalizationはPASSしたが、known-prefix finite-difference proxyは全773 wellsを`sig_r=0.004`へhigh clipし、clip fraction `1.0`でtechnical FAIL。directはparent `11.938287`から`14.062348`へ`+2.124061 ft`悪化、0/5 folds、1000+ `+2.377399 ft`、hidden-like spatial/typewell-purged `+3.278598/+3.362723 ft`、by-well p95 `+4.790247 ft`、worst `+54.818838 ft`、fixed LikPF blendも`+0.914329 ft`でscientific FAILした。proxyが量子化に支配されwell間transition noiseを識別できないことを強く示すため、decision `adaptive_sig_r_failed_close_without_rescue`としてclip/shrink/grid/`sig_p`/momentum/blend救済、inference、submission、新exp323--327相当chainなしでterminal closeした。独立兄弟`exp345_exp209_time_varying_gr_affine_calibration_hmm`の判断は変更しない。
- 2026-07-21、ユーザーの実装承認により`well_adaptive_transition_noise` / `exp309_well_adaptive_transition_noise`を、exp308未完了のまま実装だけ先行した。exp307 finite-MAD `sigma_GR`とexp308 fixed missing-distance confidenceを固定し、known-prefix rate innovationからwell別`sig_r`を作るself-contained exact-HMM trainまで実装・静的検証したが、exp307 promotion FAILによりexp308が閉じたためKaggle未実行のままterminal closeした。旧exp309はexp338の式参照元に限定し、reparent・再開しない。
- 2026-07-21 に `z_only_residual_gr_correction_ladder` / `exp321_z_only_residual_gr_correction_ladder` のKaggle private CPU Run AB version 1を611.963秒、1 diagnostic / 5 fold strata / model・booster・HMM・window decoder・control再実行各0で完了した。3,783,989 rows / 773 wells、target-free path/score freezeとlate-truth境界、finite/row/well identityはPASS。Stage AはH512 affine-quotientがZ-only `0.609237`、exp226 `0.669091`、比`0.910543`、5/5 folds、SSE説明率`0.999968`、cap4 oracle gain`3.205124 ft`でPASSした。Stage Bもtop1/top3/MRR/sign `0.332991 / 0.587903 / 0.503399 / 0.685887`がshuffleを5/5 foldsと全stress scopeで上回ったが、固定`±80 ft` bank range coverageは`0.494029`、quantization coverageは`0.604212`、最大誤差`384.734576 ft`で2固定gateをFAILした。Z-only direct RMSE`107.494824 ft`、H512 block mean residual絶対値`90.628894 ft`から、局所shapeは低次元でも固定小shift補正のscale前提が成立しないと判断する。Stage C、案4/5、inference、submissionを未実装のまま閉じ、bank/sigma/threshold/decoder救済と同系backlog追加を行わない。完了済みexp321と依存する予約案4/5をbacklogから削除し、独立した既存優先実験を維持する。
- 2026-07-21 に `tempered_raw_smoothed_exact_hmm_emission` / `exp305_tempered_raw_smoothed_exact_hmm_emission` のKaggle private CPU version 3（id_no `128079137`）を15,983.840秒、1 variant / 773 HMM well-runs / control再実行・model・booster各0で完了した。version 1はexp304 manifest field参照ミス、version 2は773/773 HMM後のsaved likPF列契約ミスで停止し、version 3では`last_known_tvt + likpf_mean_d`と全入力schemaをHMM前にfail-fast検証した。3,783,989 rows / 773 wells、finite 100%、ID mismatch 0、silent fallback 0、runtimeはPASS。一方、directは`11.938287 → 13.218199`（改善`-1.279912 ft`）、fixed likPF 50/50は`10.269693 → 10.767674`（改善`-0.497982 ft`）、ともに改善1/5 foldsで、1000+、hidden-like 2面、p95、worstも全FAILした。saved likPF baseline parityはdelta復元後に約`3e-6 ft`だけ固定許容値を超えたが、科学的悪化より十分小さくnegative decisionは不変。事前登録どおりbeta/sigma/HMM/blend救済、案3/案4、inference、submissionを行わずbranchを閉じ、完了済みexp305をbacklogから削除する。新しい同系救済は追加しない。exp305待ちを解除したexp321 Run ABも後続で完了・閉鎖済みで、独立したexp307 finite-only sigmaなど既存優先実験を維持する。
- 2026-07-21、`exp333_exp226_k16_segment_residual_offset_target`のStage 0をKaggle CPU v1でPASSし、32-well preflight v1も親OOF parity最大差`1.819e-12 ft`、full runtime外挿`1.787 h`でPASSした。canonical full train v1（id_no `128116592`）は1 variant / 1 config / 5 CPU boosters、strict nested exp226 25 fits / 3,865 prediction well-runs、control再学習0を`1,781.997 sec`で完走。CVはexp226 `9.427109597`から`9.076676661`へ`0.350432936 ft`改善し、5/5 folds、1000+、hidden-like 2面、boundary、by-well p95を改善した。一方、固定pooled上限`8.894085501`を`0.182591160 ft`超過し、near 0--250は`+0.057439 ft`、worst well `7987f2f2`は`+8.099023 ft`悪化した。pooled/near/worstの3 gate FAILで`FAIL_CLOSE_BRANCH`。exp228にも`0.132591 ft`届かず、追加config、same-OOF clip/shrink/gate救済、inference、submissionなしでbranchを閉じる。低優先の次候補は予測を変更しない0-booster failure readoutだけとする。
- 2026-07-21、ユーザー依頼によりexp295のcompute-feasible training redesign 2案を別実験として設計固定した。推奨案A `exp331_prefix_gr_unary_local_ce_exact_ssm`は、固定16-view Stage 0 T4で保守的fold外挿`4.516839 h`、peak`1.924052 GB`をPASSし、Stage A fold 0の1 neural modelをversion 1で完走した。real GRはgeometry-only `32.465002`とshuffle `57.878820`より良いRMSE`24.760360`でGR attributionを示したが、保存済みexp209 `12.671087`へ`+12.089273 ft`回帰した。well p95も`44.560719`対`26.301518`、worst regression`+63.109520 ft`で、exp209より悪化138/155 wells。runtime`4.115497 h`、peak`1.889884 GB`、truth-freeze、SHAはPASSしたが3科学gateをFAILしたため、事前契約どおりStage B、推論、提出、exp331内rescue gridなしで閉鎖した。local CEはGR信号を学ぶがglobal path品質の代替には不足すると判断する。代替案B `exp332_prefix_gr_unary_fixed_window_structured_ssm`は固定16-window T4 Stage 0 version 1（id_no `128231704`）を完走したが、peak memory `1.203263 GB`はPASS、保守的fold外挿`13.151137 h`は固定上限`8.5 h`をFAILした。事前契約どおりwindow/loss/decoder救済、Stage A/B/C、推論、提出なしで閉鎖する。exp331 local-CEは科学gate、exp332 structured-windowは計算gateで終了したため、exp295 neural unary familyの同系救済を追加しない。
- 2026-07-21、ユーザー依頼によりHMM内部の時間変化パラメータ案4--8を設計固定した。項目8は独立変数を同時変更しないため`σ_p,t`とGR affine `a_t,b_t`へ分割し、`exp323_time_varying_exp226_dip_rate_prior`、`exp324_exp226_donor_covariance_segment_sig_r`、`exp325_exp226_window_likelihood_hmm_tempering`、`exp326_residual_rate_time_varying_momentum`、`exp327_time_varying_position_sigma_floor_audit`、`exp328_time_varying_gr_affine_calibration_hmm`の6件をdesign-onlyで作成した。exp323はexp309後のP1、exp325はexp305/323後の高リスクP1、exp324/326はP2、exp327/328は低優先とする。exp325はexp321のpost-hoc TVT補正ではなくexact-HMM内のsparse observation factor、exp328は停止中exp318のType Well群priorを使わないcurrent-well causal calibrationであり、既存枝との重複を避けた。全件で実装、notebook実行、Kaggle push、推論、提出を禁止したまま、先行条件、単一変更、truth-free freeze、runtime、promotion gateを確定した。
- 2026-07-21 に `donor_support_risk_bounded_weight_shrink` / `exp329_donor_support_risk_bounded_weight_shrink` のKaggle private CPU version 2（id_no `128104811`）を209.829秒、1 risk / 1 circular control / 773 support well-runs / model・booster・decoder各0で完了した。version 1はNumPy 2でobject identity列へ`array_equal(equal_nan=True)`を使った互換性エラーで科学判定前に停止し、dtype-aware exact equalityだけを修正した。version 2は3,783,989 rows / 773 wells / 12,368 K16 segments、technical hard checksとcoverage checksを全PASSし、発火は762,529行（20.151459%）/433 wells/5 folds。一方、pooled real AUC `0.562091 < 0.60`、control AUC 0.556781との差`0.005310 < 0.05`、top-risk mean benefit `-0.674259 ft`、1000+とhidden-like 2面の方向guardもFAILした。AUC>0.5は5/5 folds、top-bottom benefit差は+0.669452 ftだが、circular controlから分離できず高risk側のdestination移動も平均悪化するため補正gateとして不採用とする。target-free contract SHAは`03049211...05f3000`。事前登録どおりthreshold/alpha/clip/destination/feature救済、Stage 1、inference、submissionなしでbranchを閉じた。完了済みexp329と、exp329 PASSを必須依存にした未実装exp330をbacklogから削除し、新しい同系救済backlogは追加しない。
- 2026-07-21 に `gr_likelihood_weak_exp226_soft_shrink_readout` / `exp322_gr_likelihood_weak_exp226_soft_shrink_readout` のKaggle private CPU version 2（id_no `128089589`）を195.332秒、1 candidate / 1 matched control / 5 exp263 strata / model・booster・decoder各0で完了した。version 1は別splitのexp226元OOF foldとexp263 readout foldの一致を誤要求してscoring前停止し、version 2ではexp263 outer foldをreadout、exp226元foldをsource監査として分離した。technical hard checks、exp263 formula parity、cached exp226 parityは全PASS。一方、発火は4,870行（0.128700%）/10 wellsで事前coverage下限1%/50 wellsをFAILし`INCONCLUSIVE_COVERAGE`。RMSEは`8.238331715 -> 8.239202313`（`+0.000870598 ft`）、activated subset `+0.688824530 ft`、改善1/5 folds、1000+ `+0.000966632 ft`、worst well `+0.261431339 ft`、real gateはcircular controlより`-0.001254155 ft`だった。coverage不足だけでなく科学guardも不支持のため、alpha/quantile/block/clip/emission/selector救済、inference、submissionなしでbranchを閉じた。完了済みexp322をbacklogから削除し、新しい救済backlogは追加しない。
- 2026-07-17 に `multi_scale_initial_rate_candidates` を `exp268_multi_scale_initial_rate_candidates` として実装した。exp209 exact HMMの保存済み`tail_n=30` controlを再生成せず、known prefixだけから`median((delta TVT_input + delta Z) / delta MD)`を固定window `32/64/128/256`で算出し、HMM grammar・Gaussian GR emission・grid・prior幅を固定した4 candidateを生成する。generatorへ渡すhorizontal frameから`TVT`を明示dropし、path凍結後にだけtrue TVTを診断へ結合する。Kaggle timeout回避のためstable SHA256で2 well shardに分け、正規trainでstrict coverage、rate spread、rate/path重複、row / 128・256・512 block / whole-well oracle、prefix長、1000+、hidden-like、worst-wellを集約する。LightGBM config / fold / boosterは0 / 0 / 0、parent/control再生成、GPU、candidate平均、selector、inference、submissionはなし。self-contained notebooks 3本、disabled inference、contract tests 6件、strict validation、private CPU package準備まで完了し、Kaggle CPU trainは未実行。元backlogは実装済みとして外し、結果が出るまでwindow候補採用、add-only feature化、raw-test portは判断しない。
- 2026-07-17 に `pf_ancc_small_seed_mean_candidate_audit` / `exp271_pf_ancc_small_seed_mean_candidate_audit` のKaggle CPU train version 2を完了した。version 1は全773 wellsのPF生成後、exp072 float32 `target + last_known_tvt`復元値とexp266 raw TVTの精度差でmean4 per-well parityが最大0.000459 ftずれてfail-closed停止した。許容値を緩めず、candidate gzip保存後にraw TVTをfloat64で評価へjoinする修正だけを入れたversion 2は3,783,989 rows / 773 wells、600 particles × 固定8 seeds、runtime 1,386.570秒、LightGBM config / fold / booster 0 / 0 / 0で完走した。seed0はexp072へ全行差0、mean4/mean8 per-well RMSEはexp266へ最大7.105e-15 ft差、exp263 manifestと60 partition SHAもPASS。standalone RMSEはseed0 14.493051、mean4 13.126896、mean8 13.027107。core12へのoracle deltaはrowでmean4 -0.046543 / mean8 -0.049720 / both -0.065252、whole-wellで-0.028392 / -0.036973 / -0.050751。row unique-bestはmean4単独252,772（6.6800%）、mean8単独251,635（6.6500%）、両方追加時合計340,687（9.0034%）だった。仮説は支持し、単一candidateならmean8単独headroomの約93〜95%を半seedで回収するmean4へ縮約する。一方、両候補には相補性があるため、保存済みpathを使う次のadd-only selector監査ではmean4/mean8とseed disagreementを残す。raw-test PF再生成、hard oracle routing、inference、submissionは進めない。
- 2026-07-18 の `exp276_exp264_compact_tail_risk_target_free_gate_audit` は計算上完走したが、その後、入力のexp264 Stage C compactとStage D add-only OOFがfeature availability leakageで無効と判明した。したがってexp276のrisk lift、gated RMSE、quantile guard、negative resultも全無効化し、性能・backlog判断に使用しない。
- 2026-07-21 に同じ `exp276_exp264_compact_tail_risk_target_free_gate_audit` をcorrected exp264 Stage C v6 / Stage D v3へ入力だけ差し替え、Kaggle private CPU version 3（id_no `127735777`）で有効に再検証した。1 audit、5 evaluation folds、LightGBM config / trained fold / booster `0/0/0`、104.017秒。technical contractは全PASSしたが、q70/q80/q90の固定guardは全FAILした。q90はpooled lift `1.165139 / 1.211019`と改善保持`59.77%`を示した一方、positive-lift foldsは`2/5 / 4/5`、worst-wellは`+13.441268 ft`で、事前条件を満たさない。q70/q80も改善保持50%未満であり、quantileやfeatureの事後救済、inference、submissionを行わずbranchを閉じた。これによりexp303の`exp276_completed` / `exp276_promotion_guard_fail` dependencyは成立した。
- 2026-07-21 に `exp226_multiscale_k_stability_selectability_readout_on_exp264` / `exp303_exp226_multiscale_k_stability_selectability_readout_on_exp264` のKaggle private CPU version 1（id_no `128080983`）を約142.125秒、1 fixed readout × 5 evaluation folds、model / trained fold / booster `0/0/0`で完了した。feature coverage、duplicate block 0、truth-before-freeze 0、固定input SHA、score再計算は全PASS。一方、7,787 H512 blocksのpooled AUCは`0.488805`、AUC>0.5は`1/5 folds`、top/bottom positive-rate liftは`0.916190x`、mean K16 benefit差は`-1.205532 ft`、1000+ / hidden-like方向は`0/3`で全scientific guard FAILとなった。高instability側ほどK16 benefitが低い逆方向が4/5 foldsと全stress scopeで再現したため、K-scale instabilityはK16 misrankingのtarget-free selector signalにならないと判断する。完了済みexp303をbacklogから削除し、方向反転、feature weight/horizon/boundary幅/threshold救済、selector学習、inference、submissionは行わない。新しい救済backlogは追加せず、独立したexp305 exact-HMM emission auditを優先する。
- 2026-07-18 に `exp277_pf_ancc_small_seed_mean_addonly_selector_audit` のmean4 selector version 1（id_no `127737879`）はCPU 40 boosters、約6,984.277秒で技術的に完走したが、training-only formation raw/delta 12特徴のため無効化・quarantineした。2026-07-19に修正版exp264 Stage A v4 / Stage C v6のraw-test-only 88特徴へportし、ユーザー再承認でcorrected mean4 1 variant × 2 objectives × outer 5 × inner 4 = 40 CPU modelsをcanonical version 2として5,707.598秒で完了した。candidateは旧`pf_ancc`を残さず同じslotへ`pf_ancc_seed_mean_4`を置換、親schemaとの差はcandidate identity 1列だけ。`MD/X/Y/Z/GR`はtrain 773/773・current-test 3/3で存在し、formation raw/delta hit 0。40 model実体SHA、model/compact manifest、25 partition契約、nested leakage guardはPASS。selector scoreはexpected-error MAE `3.793764`対prior `5.708749`、within10 logloss `0.360024`対`0.509003`、Brier `0.112272`対`0.164560`で全指標pooled + 5/5 folds PASSした。一方、corrected exp264 original `pf_ancc`比はexpected-error `-0.005055` / 4-of-5改善、logloss `+0.000612` / 2-of-5、Brier `+0.000441` / 2-of-5でmixed。mean4 compactは有効なdownstream入力候補とするが一様優位は主張せず、TVT downstream、mean8/both、PF再生成、推論、提出は未実行。local run gateはfalse、旧version 1と旧380列controlは使用禁止を維持する。
- 2026-07-18 に `formation_gradient_prefix_stability_risk_readout_on_exp273` / `exp278_formation_gradient_prefix_stability_risk_readout_on_exp273` のKaggle CPU version 2を0 boosterで完了し、negative resultとして閉じた。version 1はruntime判定不備でreadout cellをskipしたtechnical no-opで、判定だけを修正したversion 2は3,783,989 rows / 773 wells、full-valid 111 wellsを約49.6秒で完走した。exp273 5 candidate x 773 wellsのRMSE parity 3,865件とfull-plane parity、入力/artifact SHAは全PASS。full / last-512 / last-256のangle、magnitude、plane RMSE、rank、condition、validityをpair最大・等重みしたtarget-free riskは、primary pooled Spearman `0.074245`、q0 / q4 mean bank delta RMSE `-2.195778 / +2.157694 ft`でpooledと両端quintile guardは通ったが、fold Spearmanは`0.059649 / 0.177444 / 0.125889 / -0.123478 / -0.061654`で正方向3/5となり、必須5/5をFAILした。candidate別とbank-maxも同じfold符号反転を示すため救済根拠にしない。0 variant / 0 config / 0 trained fold / 0 booster / HMM生成0、inference/submission disabledを維持し、component/window/clip/weight/threshold grid、別gate、HMM再実行、raw-test inference、submitへ進めずexp273 formation-gradient branchを閉じる。新規救済backlogは追加しない。
- 2026-07-13 に `adaptive_likelihood_pf_trajectory_containment_audit` / `exp241_adaptive_likelihood_pf_trajectory_containment_audit` を部分監査で閉じた。Kaggle CPU shard 0/2/3の574/773 wells、2,813,393 rowsでは、gated T=2はpaired regenerated T=1にoverall `-0.011971`、1000_plus `-0.012604`と微改善したが、hidden-likeは約`+0.011～0.012`、worst-well最大`+1.666805`。mean absolute path divergenceは8 rows `0.127775 ft`からend `3.123176 ft`へ増え、trajectory containmentは支持されなかった。保存済みexp072との差約+1.9はT=2固有ではなくpaired replay parityが主因。ユーザー判断によりshard 1とstrict four-shard mergeを省略し、direct robust likelihood、追加temperature/mixture/process-noise grid、raw-test inference、submitは不採用として閉じる。
- 2026-07-13 に `fixed_lag_particle_smoother_pf` / exp235 の exact 4-shard lag64 audit を閉鎖した。3,783,989 rows / 773 wells の strict ID/well merge で `pf_lag64_mean` RMSE 13.495448、exp072 `likpf_mean` 11.594898（+1.900550）、within10 0.673755（-0.099053）となり、全距離帯で悪化した。特に `1000_plus` は +2.035222。ユーザー判断により lag128/256、seed-paired 再監査、raw-test inference、submit は行わない。exp235 と frozen exp072 は particle seed policy が異なるため smoothing 単独の因果量は未分離だが、実装candidateの採用不可は確定する。HMM は既に full forward-backward smoothing を行うため、単なる fixed-lag HMM は新規情報追加ではなく既存HMMの近似であり、現時点で backlog へ追加しない。
- 現在のアンサンブル route Public LB anchor は `exp082_public_artifact_replay_followup` ref `53885305`、Public LB 7.601。ML + PF/Beamの直接blend / public notebook replayとして管理し、`ml_model`へ再分類したexp264 / exp287とは分離する。旧 PF route / 公開 notebook再現基準は`exp027_public_replay_needless090_sel15_spread3`の8.781。
- 現在のMLルートraw deterministic anchorは`exp073_gpu_reproducibility_guard_for_exp063_full_replay`のCV 9.526374749 / Public LB 8.780。Public-LB submitted anchorは`exp287_fold_safe_formation_74_addonly_on_exp264` ref `54842141` / 7.530で、直前exp264 ref `54818932` / 7.562を-0.032、その前のexp274 ref `54793316` / 7.715を-0.185改善した。ただしexp287はworst-well +8.228410 ftと悪化well数増加、exp264はworst-well +14.482873 ftでguard FAILのため、LB anchor更新をtrain-side採用とは扱わない。exp287 / exp264はPF/HMM/Beam候補を補助compact meta featureとして使うが、direct blendやhard-pathを行わずdownstream LightGBMが最終予測を生成するためML routeとする。
- pseudo-tail 自前系は `exp061_seedbag_anchor_model_diff_distance_gate` の Public LB 11.826 が到達点。exp063 との差が大きいため、微調整は主戦場にしない。

### 固定化した失敗パターン

- PF/Beam 候補値の直接提出、hard switch、row-wise hard replacement は採用しない。使う場合は selector / confidence feature / no-training diagnostic に限定する。
- 公開 sample output の copy、public visible well だけで成立する wrapper、source artifact 依存の提出は採用しない。hidden rerun で生成ロジック自体が動く notebook だけを候補にする。
- GR / typewell / spatial prior は raw TVT 候補や後段補正として直接入れない。fold-safe に作った quality、disagreement、confidence、candidate score として評価する。
- target 変更、U-space、row-step delta、sequence residual は小規模 ablation で反証してから広げる。near row、worst-well、hidden-like stress、cumulative drift を壊す場合は global CV 改善でも進めない。
- stochastic PF/Beam / GPU 生成物は deterministic anchor にしない。feature content SHA、prediction SHA、submission SHA、kernel version が揃うまで比較候補扱いに留める。
- 2026-07-13 の `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218` は、strict nested selector 20 boostersと、ユーザー承認によるfinal GPU 15 boostersを完走した。selectorはglobal -3.089911、near `000_050` -0.609540、`1000_plus` -3.372225改善したが、worst-well最大回帰+37.680897でguard不通過。final nested OOF `lgb_mean`は7.936690でexp218 8.475794より名目-0.539104だが、selector artifact outer foldsがhistorical exp218およびGPU runtime再構成GroupKFoldと一致しないため、特徴追加の因果差とは扱わない。提出ref `54647064` はpublic-test row artifact依存で失敗したが、行依存特徴とouter別selector scoreをcurrent testから単一notebook内で再生成するhidden-safe v3へ修正した。再提出ref `54662073`はPublic LB 7.775で完了し、exp218 ML anchor 7.843を-0.068改善したためML route submitted anchorをexp238へ更新する。ensemble route anchorはexp082 7.601を維持する。後続の184 context finite copcf parity版ref `54725625`はPublic LB 7.842で旧refより+0.067悪化したため、NaN修正の実装健全性だけを採用し予測variantは不採用とする。同一fold base-only attributionには追加GPU 15 boostersの承認が必要で、再訪時は先にouter-fold safeなwell-riskまたはcandidate family exclusionのselector-only auditを優先する。

### 判断メモ補足

- 2026-07-28 に `roughening_x10_likpf_full_oof_ablation` / `exp416_roughening_x10_likpf_full_oof_ablation` をterminal closeした。Kaggle CPU 4 shardは3,783,989 rows / 773 wells / 98,944 trajectoriesを完走し、strict merge version 2でcandidate RMSE `13.617717558`、saved exp072 control `11.594894396`、regression `+2.022823162 ft`。5/5 folds、raw/missing、1000+、hidden-like 2面がすべて悪化し、by-well p95は`+14.104742 ft`、worstは`+41.050361 ft`だった。一方、exp410固定16 persistent-offset episodesのSSEは`24.700364%`改善し、局所的なbasin recoveryは再現したがglobal roughening増加へ一般化しなかった。scientific AND gateをFAILし、exp209 row parityも`0.000471875 ft`対上限`0.00001 ft`でtechnical FAIL。roughening倍率や関連parameterの救済、probe、inference、submissionは行わない。保存生成物による0-PF readoutはexp422で完了し、下記のとおり事前固定したtarget-free regimeも反証された。
- 2026-07-28 に `roughening_x10_failure_regime_attribution_readout` / `exp422_roughening_x10_failure_regime_attribution_readout` のKaggle private CPU version 2（id_no `128921651`）を完了した。technical gateはPASSし、truth / control / by-well / episode outcomeを開く前に773 wellsと3,783,989 row scopeをfreeze、入力SHA・pooled/by-well parity・0 PF/model/booster/GPU実行量を確認した。scientific gateはFAIL。recovery-pressureは期待した正方向と逆のrho `-0.166698`、positive folds `0/5`、one-sided p `1.0`、damage-exposureはrho `-0.041485` / p `0.111301`。固定high-recovery / low-exposure cellもrow RMSE gain `-1.852450 ft`、改善`1/5` folds、rest比equal-well gain `-0.518966 ft`だった。persistent-offset 4 episodesのSSEは`45.801967%`改善したが、全positive episode gainのshareは`39.400617%`で50%閾値に未達。exp416のFAILを維持し、score / threshold / cell / roughening parameter救済、adaptive policy、inference、submissionは行わず、attribution branchもterminal closeする。実装済みbacklogは削除する。
- 2026-07-20 に `prefix_anchored_wholewell_gr_alignment_ssm` / `exp295_prefix_anchored_wholewell_gr_alignment_ssm` のStage Aをruntime FAILで閉じた。version 2はhard truth path infeasible、Gaussian soft-label structured NLLへ単一修復したversion 3は最終status `CANCEL_ACKNOWLEDGED`で、18.456秒のcontract preview以降にepoch summaryを出せずKaggle timeoutした。fit 556 wells / 1,668 views / 8,571,405 suffix rows、平均571.697 position states × 41 rate states、通常/label-conditioned posteriorの4 DP sweepsにより、1 epochだけで`803,449,626,924` position-rate cells、最大8 epochsで`6,427,597,015,392` cellsだった。model/epoch/outputは0で精度仮説は未評価だが、固定8.5時間runtime gateはFAIL。exp295内のview/epoch/loss/sigma/band/architecture救済、version 4、Stage B/C、inference、submissionは行わない。実装済みbacklogは削除し、新規expを選ぶ場合だけ低優先のcompute-feasible training redesign候補へ分離する。現行優先度はexp303閉鎖後もexp305 tempered exact-HMM候補をこの再設計より上に置く。
- 古い自前 PF/Beam、DTW/DWT、hard router、prefix online training、Ravaghi single-model parity、exp031/033/035 型の public sel15 後段補正は、現時点では再投入しない。必要な場合だけ失敗パターンや比較基準として参照する。
- exp063 の Pixiux likelihood-PF diagnostics は古い自前 PF/Beam とは別物として扱う。通常の ML 改良では直接置換ではなく、prefix backtest、residual clip、sample weight、typewell / spatial prior の材料に限定する。`pf_beam_disagreement_error_map` は完了済みで、発見は `docs/surveys/pf_beam_disagreement_error_map_20260620.md`、生の表は`studies/pf_beam_disagreement_error_map/`を正とする。
- exp069 で Pixiux `likpf_mean` direct PF/Beam submission を実施した。pre-patch v2 は Public LB 9.877 (`ref=53637978`) で悪化し、deterministic patch 後の v3 も Public LB 9.721 (`ref=53706005`; duplicate `ref=53705994` も 9.721) で exp063 v2/best 8.811 より +0.910、exp027 8.781 より +0.940 悪化した。`ref=53710264` / `ref=53710105` の Public LB 8.766 は exp069 への紐づけ誤りとして扱う。LightGBM を通さない直接提出は採用しない。
- exp068 で exp039/exp038 系 CV surface 上に exp063 tracker/PF/Beam output features を載せて Pixiux LightGBM family を再学習評価した。`lgb_mean` は original-fold 11.878856 / well-hash 11.994729 で、exp039 型 surface 上では旧 single-LGBM より強いが、exp063 strict public replay CV 9.630105 とは評価面が違うため anchor 更新根拠にはしない。提出 ref `53654439` は Public LB 762.715 だったが、旧 inference が静的な exp063 public-sample prediction artifact に依存しており hidden scoring で fallback した可能性が高いため採用しない。2026-06-16 のユーザー指示により exp068 は破棄し、同じ目的は対象を exp063 から exp073 に差し替えた backlog `exp073_exp039_cv_reassessment` として作り直す。
- exp076 で exp068 の作り直しとして exp073 full replay LightGBM family を exp039 CV surface で再評価した。train v3 は `leave_one_original_fold_out/lgb_mean` 9.696040174、`well_hash_holdout/lgb_mean` 9.553554167。inference v1 は raw test から PF/Beam/likelihood-PF features を再生成した。提出 ref `53757190` は Public LB 8.799 で、exp027 8.781 / 近接 exp073 8.780 より悪化したため採用しない。exp039 CV surface reassessment は完了として閉じる。
- exp077 として `exp073_full_replay_postprocess_guard` の Kaggle train v1 / inference v1 を完了した。実験番号は最新 `exp076` の次として `exp077_full_replay_postprocess_guard` を使った。same-OOF best は `longtail_likpf_tiny_gate_w006` で RMSE 9.470514772、exp073 baseline 9.526374749 から -0.055859978 改善。固定 policy inference は raw test PF/Beam/likelihood-PF replay features を再生成し、submit-check PASS。manual submission ref `53809333` は Public LB 8.611 で exp073 Public LB 8.780 を改善し、当時の ML route submitted/postprocessed anchor になった。2026-06-22 に exp092 が Public LB 8.350、exp098 が Public LB 8.441 で上回ったため、現在は直前 anchor / 比較基準として保持する。exp073 は raw deterministic anchor として保持する。
- exp065 の native row-lag overlap group は typewell / spatial prior の入力候補として有望。validation well 自身や同 fold valid の true TVT を neighbor pool に入れないことを最優先の制約にする。
- exp175 で exp148/exp092 ML output への cluster-outlier gated prior direct correction は不採用とした。続く exp181 で exp109/114 と同じ PF/Beam/likPF 候補への gated correction を検証し、best gated は `likpf_mean` RMSE 11.594897672 -> 11.479140438 と改善、exp109 global reference の worst-well regression +6.594183 を +4.359666 まで下げた。ただし direct posthoc correction としてはまだ worst-well regression が大きいため、PF/Beam 候補への cluster-outlier prior correction も inference port / submit には進めない。再利用する場合は selector / confidence feature / candidate scoring の材料に限定する。
- `cluster_outlier_prior_confidence_addonly_on_exp158_selector` は exp183 として Kaggle train v2 を完了。exp157/158 の 8候補 selector に exp181 cluster-outlier typewell/spatial prior signal を add-only score feature として追加し、3 LightGBM configs x 5 folds = 15 boosters と exp158 同一 Viterbi 180 variants を評価した。best Viterbi は RMSE 10.601481774 / MAE 6.386571251 / within10 0.792418794 で、exp158 continuity RMSE 10.789163253 から -0.187681479 改善し、path switches は 5,650 / 1.493 per 1000 rows。direct correction ではなく selector confidence feature としては train-side supported。ただし v2 は OOM 対策で long-model train/eval cap を 120k/fold にしているため、inference port / submit 前に raw-test parity、worst-well / bucket / exp115 subgroup、必要なら高メモリまたは split train を同じ exp183 内で確認する。
- `cnn_sdf_mtp_heatmap_path_features_on_exp158` は exp184 として Kaggle train v2 を完了。v1 は fold0 multiclass 後に `DeadKernelError`、v2 は exp183 と同じ long-model memory guard に寄せて完走した。best Viterbi は RMSE 10.560650325 / MAE 6.329187986 / within10 0.797056492、exp158 continuity RMSE 10.789163253 から -0.228512928 改善し、path switches は 5,713 / 1.509782 per 1000 rows。selection は `likpf_mean` 42.25%、`pf_ancc` 38.26%、dense family 15.13%。heatmap candidate-distance features が feature importance 上位に入り、add-only heatmap path features は train-side supported。ただし heatmap sparse distance q4 は RMSE 14.058409、exp115 spatial valid は 12.696140、typewell purged valid は 12.629861 と hidden-like / sparse-coverage stress が残る。heatmap direct TVT replacement / softmax weighted TVT / PF weight replacement は行わず、次は同じ exp184 内で raw-test heatmap feature generation、sparse interpolation coverage、feature schema parity、fallback behavior を確認してから inference/submit 可否を判断する。
- `exp184_heatmap_selector_compact_addonly_on_exp148` は 2026-07-05 に Kaggle CPU split train v1 を完了し、train-side rejected。`train_lgb0` / `train_lgb1` / `train_lgb2` はすべて `COMPLETE`、各 1 LightGBM config x 5 folds、合計 15 boosters、control / parent 再学習なし。split CV は lgb0 8.710685277、lgb1 8.639432353、lgb2 8.611075285、OOF 3本を chunked streaming で結合した cross-split `lgb_mean` は 8.604130846。exp148 GPU historical `lgb_mean` 8.501281182 から +0.102849664、exp148 CPU runtime `lgb_mean` 8.528698114 から +0.075432732 悪化し、exp188 add-only 8.539573790 も下回ったため inference / submit はしない。Kaggle split train input に exp148 train output がなかったため optional exp148 OOF delta features は unavailable で、local smoke の 31 features ではなく 28 hmp184 features で完了した点は caveat として残す。厳密な 31-feature rerun は可能だが、現時点では優先しない。backlog は完了/不採用として外す。
- `cnn_pf_likelihood_probe` は exp197 として Kaggle train v1 を完了。exp099 fixed PF/Beam/likPF candidate cache と raw train local GR/typewell window から candidate-level CNN/SDF likelihood scorer を学習し、point-GR likelihood、exp099 multiobs score、exp111 learned likelihood、likPF baseline、shuffled/no-GR negative control と比較した train-side GPU diagnostic。real_gr learned_prob candidate AUC は 0.908691639、shuffled_gr 0.902727327、no_gr 0.905303044 で、real GR の上積みは shuffled に +0.005964、no-GR に +0.003389 と小さい。top1 learned_prob RMSE は 11.301053 で likPF single 11.293248 よりわずかに悪く、exp111 learned probability AUC 0.915825 も下回った。decision は `weak_real_gr_signal_needs_guarded_followup`。PF weight replacement、PF/Beam 再生成、raw-test feature generation、submit へは進めない。追加で確認するなら candidate scalar / row context を制限した GR-only ablation だが、現時点では低優先。
- `heatmap_mdn_candidate_generator_probe` は exp202 として Kaggle train v1 を完了。heatmap only top10 は within10 0.808907780 / oracle RMSE 13.352563025 で existing PF/Beam union より弱いが、existing + heatmap top10 は oracle RMSE 5.068679053 -> 2.745528140、within10 0.949639623 -> 0.986970985、new-best candidate rate 0.252541120 と大きく positive。`1000_plus` bucket は 6.413572416 -> 3.295946470、by-well は 668 improved / 105 same / 0 worse。候補集合 headroom は支持されたが、oracle readout なので direct TVT replacement、softmax average、PF weight replacement、submit はしない。後続の exp203 feature-only、exp207/208/210/212/215 path artifact probes まで確認した結果、生成 path 自体が弱いため heatmap 由来 path 生成 route は closed/rejected とした。
- `heatmap_mdn_addonly_selector_or_ml_features` は exp203 として Kaggle train v1 を完了。exp202 heatmap MDN topK を selectable candidate にはせず、既存 exp184 selector の 8 候補に対する add-only `hmdn_` confidence / distance features として追加した。best Viterbi は RMSE 10.665741318 / MAE 6.350286735 / within10 0.797977743、path switches 12,807 / 3.384524 per 1000 rows。exp158 continuity 10.789163253 からは -0.123421935 改善したが、exp184 best 10.560650325 からは +0.105090994 悪化し、path switch も exp184 より増えた。feature-only signal は確認できたが exp184 を更新しないため inference / submit はしない。後続 exp202/207/208/210/212/215 で oracle headroom と full-grid artifact は確認したが、生成 path 自体が弱いため、heatmap 由来 path 生成 route はユーザー判断で closed/rejected とし、exp204 系 selector candidate follow-up は行わない。
- `heatmap_mdn_overlapping_window_path_stitch_probe` は exp207 として Kaggle train v2 を完了。v1 は notebook kernelspec metadata 不足で `No kernel name found` 失敗、v2 で Python 3 kernelspec を追加して完走した。exp202 v2 local path artifact を well 内で target-free beam stitch し、exp099 candidate cache covered rows 1,333,241 / coverage 0.352337441 上で oracle readout を確認した。existing union oracle RMSE 5.154353660 に対し、existing + stitched top1 は 4.472998031、top3 は 4.418699605、top3 new-best rate 0.069157039。`1000_plus` は 6.376418 -> 5.414525、by-well は 461 improved / 312 same / 0 worse。ただし stitched only top3 は RMSE 50.798377042 と粗く、source overlap は 773 wells 中 3 wells / 39 center pairs だけで、現行 exp202 artifact は full-well overlapping stitch の証拠として不十分。direct replacement、softmax average、PF weight replacement、inference、submit はしない。この backlog は完了として外し、dense stride follow-up は exp208 で完了した。
- `heatmap_mdn_dense_stride_window_path_regeneration_probe` は exp208 として Kaggle train v1 を完了。exp202 saved fold model から stride 64 dense local paths を再生成し、source overlap は exp207 の 3 wells / 39 pairs から 773 wells / 24,679 pairs、row coverage は 0.352337441 -> 0.430091631 に増えた。一方、local topK10 existing + stitched top3 oracle RMSE は 4.420752853 で exp207 の 4.418699605 を更新せず、stitched only top3 も 47.188322489 と弱い。dense path が物理的に stitch 可能なことは確認できたが、direct replacement / softmax average / PF weight replacement / inference / submit はしない。exp210 で covered-row contract 化、exp212 で full-grid contract 化、exp215 で fallback 0.0 artifact まで完了したが、生成 path 自体が弱いため heatmap 由来 path 生成 route は closed/rejected とした。
- `heatmap_mdn_full_well_path_generation_probe` は exp210 として Kaggle train v1 を完了。exp208 dense path `.npz` / samples を読み、exp207/208 と同じ target-free stitch score で local topK 5/10 から exp099 candidate-cache intersection の selector-facing covered-row contract `well,row_id,md_from_ps,path_rank,tvt_pred,source_window_count,overlap_weight,assignment_gap_flag,local_rank_mix,path_step_abs,curvature_abs,candidate_score` を保存した。primary local topK10 contract は required columns present、duplicate key rows 0、null required values 0、rows 8,137,310、unique row ids 1,627,462、coverage 0.430091631、source overlap 773 wells / 24,679 pairs、source gap 0。existing + stitched top5 oracle RMSE は 5.139413349 -> 4.407737500、new-best rate 0.075083781、by-well は 524 improved / 249 same / 0 worse、`1000_plus` は 6.352451 -> 5.403899。artifact contract は成立したため backlog は完了として外す。一方、artifact は exp083/exp072 の全 `md_since` 区間を覆う full trajectory ではなく、stitched-only top5 も RMSE 46.958946049 と弱いため、direct replacement / softmax average / PF weight replacement / inference / submit はしない。後続 exp212/215 でも生成 path 自体の弱さが残ったため、heatmap 由来 path 生成 route は closed/rejected とし、selector の通常候補化は行わない。
- `exp212_heatmap_mdn_full_grid_path_generation_probe` は Kaggle train v1 を完了。exp208 cached dense paths を target-free stitch した sparse source rows から、exp099 feature-cache row grid に rank1-5 full-grid path artifact を生成した。contract は rows 18,919,945、unique row ids 3,783,989、row coverage 1.0、duplicate key rows 0、null required values 0 で成立。一方、source coverage は 0.430091631、fallback unique row rate は 0.569908369 で、実 run の fallback は right extrapolated。plot overlay で途中から直線 tail になるのは、exp208 source が `max_tail_rows=2048` までの dense windows に限定され、exp212 がその後ろを endpoint 外挿しているため。existing + stitched top5 oracle RMSE は 7.434029841 -> 5.941479995、by-well は 567 improved / 206 same / 0 worse と headroom があるが、stitched-only top5 RMSE は 50.085237573 と弱い。direct replacement / softmax average / PF weight replacement / inference / submit はしない。exp215 で learned `path_logit` を持つ MTP full-tail artifact が coverage 1.0 / fallback 0.0 を達成したが、heatmap 由来 path 生成 route は後続判断で closed/rejected とした。
- `exp215_mtp_full_tail_heatmap_path_generator_probe` は Kaggle train v1 / T4 GPU を完了。exp202 由来の 5ch heatmap input を維持し、`path_pred [K,L]` と learned `path_logit [K]` を出す MTP full-tail generator を 5 folds / 5 CNN models で学習した。full-grid contract は rows 18,919,945、unique row ids 3,783,989、wells 773、row coverage 1.0、fallback unique row rate 0.0、duplicate key rows 0、null required values 0 で成立し、exp212 の fallback-heavy / endpoint hold tail 問題は解消した。既存 PF/Beam union oracle RMSE 7.434029932 に対し、existing + learned MTP top5 は 5.113654814、within10 は 0.906525363 -> 0.945863743 と改善。一方、learned MTP top5 only は RMSE 32.333142886、weighted path は 59.272141581 と弱いため direct replacement / softmax weighted TVT / PF weight replacement / inference / submit はしない。fallback 0.0 でも生成 path 自体が弱いため、heatmap 由来 path 生成 route はユーザー判断で closed/rejected とし、exp204 系 topK candidate / confidence feature follow-up は行わない。
- `discussion711308_dz_dtvt_bpeak_cluster_baseline` は exp206 として Kaggle train / inference / code submission を v1-v4 で完了した。v1 は `dTVT/dMD ~= a*dZ/dMD+b` の rate-fit と exact typewell / b-peak / XY nearest assignment で、ref `54395246` Public LB 41.214。v2 は discussion 本文に寄せた row-step `dTVT ~= a*dZ+b`、X/Y/Z + last-300 TVT/Z feature-nearest、visible prefix holdout source selector で、train best `prefix_holdout_source_b_fixeda_h600` RMSE 35.410555130、ref `54396544` Public LB 34.908。v3 は full X/Y/Z geometry と last-300 TVT/XYZ shape samples の deterministic cluster、cluster/local source `a,b` を使う `discussion_fullxyz_cluster_holdout_ab_k24_h300` で、train CV RMSE 35.300417350、ref `54408573` Public LB 29.193。2026-07-08 にユーザー指定で v4 `known_tvt_fit_full` を追加し、source / cluster `a,b` を選ばず、query/test well 自身の known `TVT_input` 全体で `dTVT ~= a*dZ+b` を fit して unknown suffix を累積予測した。v4 は train CV RMSE 52.507422925、ref `54458212` Public LB 57.063 で v3 より悪化し、要件 LB 約 12.8 も大きく未達。direct known-TVT fit 経路は採用しない。
- 2026-06-19 に `exp079_public_artifact_replay_integrity_audit` v4 を Kaggle で完了した。Kernel は `kentookumura/exp079-public-artifact-audit-train` v4、output は `/tmp/kaggle-output/exp079_public_artifact_replay_integrity_audit/train_v4`。Pilkwang / ridge-sp の mounted sources は確認済み、Missing sources 0、notebook inspections 2、candidate files 28、valid submission CSV 17、pairwise distance 136。Pilkwang final `submission.csv` は `submission_projected_ridge_pf_pretrained_lgbm_base.csv` / `w0.55` と同一。Pilkwang final vs projected ridge/PF projection は RMSE 1.299277767、vs pretrained LGBM は 1.588006160、vs model-package-only は 17.318521442、vs ridge-sp final は 2.020019968。Pilkwang notebook は `exact_match_or_override=38` / `writes_submission_csv=3` の risk hits があるが、exact-match recovery / guarded overlap override は現設定では無効。ridge-sp / SP45 / fle3n / Koolbox / LB 7.776 系も次点候補だが、SP45 / fle3n / Koolbox は exact source slug 固定が未完了。
- 2026-06-19 に `exp081_pilkwang_branch_decomposition` を実装し、exp079 v4 の summary / submission summary / pairwise JSONL から Pilkwang branch 分解を完了した。候補 16 件はすべて submission contract valid。shortlist は 6 件、submit 検討候補は 2 件に絞った。rank 1 は `submission_projected_ridge_pf_projection_d4_b075_raw.csv` で vs final RMSE 1.4422981136 / vs ridge-sp RMSE 1.1301896874。rank 2 は `submission_projected_ridge_pf_pretrained_lgbm_w0.60.csv` で vs final RMSE 0.1443641963 / vs ridge-sp RMSE 1.9410101494。pretrained LGBM 単独は vs ridge-sp RMSE 3.2053172172、model-package-only は vs final RMSE 17.3185214417 で提出候補から外す。exp079 v4 local output は候補 CSV 本体を保存していないため Pilkwang branch の row-level guard は未実施。exp027 / exp073 / exp063 anchor との pairwise も未保存で `missing_pairwise` として扱う。
- 2026-06-19 に `exp082_public_artifact_replay_followup` v2 と追加 guard を完了した。mountable source の audit は `audit_completed`、missing required sources 0、candidate files 19、valid submission CSVs 18、source inspections 7、pairwise distances 153。SP45 projection 3 件は submit-check PASS。fle3n vs jaemin は RMSE 0.324981626、p95 abs 0.618058119 と近い。ridge-sp との差は fle3n RMSE 1.384232857、jaemin RMSE 1.413346840、rauff direct-output RMSE 1.303650505。`rauffauzanrambe/rogii-sp45-wellbore-for-blend-prediction` は Kaggle source として mount できないため code-submit 再現候補にはしない。2026-06-20 JST に fle3n SP45 projection の public output copy wrapper ref `53853237` を提出したが hidden rerun error になった。その後、`fleongg/fle3n-rogii-v4` の Engine A / SP45 projection 生成ロジックを source-port した ref `53854058` を提出し、Public LB 7.857 を記録した。`sp45_fleongg_source_port_next_candidates` では fle3n final SHA `359b3e77...` と jaemin final SHA `d8b0af2c...` が source risk 上 next source-port run 候補、Pilkwang raw projection / w0.60 は exact archived source missing と判定した。fle3n final source-port run は `kentookumura/exp082-fle3n-final-source-infer` v1 で完了し、submission SHA `40ffcd3d...`、submit-check PASS。public fle3n final との差は RMSE 0.292760267、previous exp082 SP45 source-port との差は RMSE 1.665882481。提出 ref `53885305` は Public LB 7.601 で、exp082 SP45 source-port 7.857 から -0.256 改善した。
- 2026-06-25 に `exp122_exact_override_negative_control` の Kaggle train v1 を完了した。Kernel は `kentookumura/exp122-exact-override-negative-control-train` v1、output は `experiments/exp122_exact_override_negative_control/kaggle/output/train_v1`。decision は `negative_control_passed_current_evidence`。Pilkwang final は archived base branch と一致し、exp064 hidden code submission は exposed filename-prefix の train/test well_id overlap assertion non-trigger。guard output summary は見つからず、`guard_changed=false` / `guard_rows=0`。archived notebook source では same-well shortcut flags が enabled として見える一方、exp079 source spec は exact/override disabled check を期待しているため、この矛盾は hidden-safe 改善根拠ではなく risk として記録する。同じ物理 well が別 anonymized id で出る可能性は否定しない。結論として same-well exact / guarded override は改善根拠にも submit 候補にも採用しない。
- exp063 は GPU rerun で train-side CV が bitwise 再現しなかった。Public LB 8.811 の提出物は詳細ログ側で固定できているが、CV は単発 GPU 実行値として扱う。exp070 は 65-feature compact tracker surface で実装・実行してしまったため、exp063 full replay 再現性監査としては破棄する。一方で、2026-06-14 のユーザー報告では exp070 が Public LB 記録を更新した。Kaggle submissions には近接して `8.548` (`ref=53669416`) と `8.515` (`ref=53669453`) があり、提出 description が空のため ref と生成物の紐づけ確認が必要だが、exp070 は `invalid_as_repro_guard_valid_as_lb_candidate` として再分類する。exp074 で同じ 65-feature compact surface を明示的な LB candidate audit として再実行し、Kaggle train v1 / inference v1 を完了した。CV は `lgb_mean` 9.731506199、submit-check は PASS。Public LB は未提出なので anchor 昇格はしない。exp075 は exp070/exp074 compact PF/Beam surface の再現性を担保する実装として位置付ける。CPU 専用 PF/Beam feature generation v4、LightGBM train v2、GPU inference v3 を完了し、GPU inference v3 は Public LB 8.489、submit-check PASS。確認済み exp075 提出は `ref=53807892` / `ref=53807896` で、最新 `ref=53809333` は exp075 ではない。CPU inference v2 と GPU inference v3 は prediction/submission が一致したが、regenerated test feature content は runtime mode で異なったため、feature content の厳密比較は同一 runtime mode で行う。したがって後続の compact surface 利用は exp070/exp074 ではなく exp075 の生成物・notebook 構成を正とする。exp072 v2 で train cache generation の PF/Beam/likelihood-PF を stable per-well seed 化し、exp073 で LightGBM train と deterministic raw-test PF/Beam regeneration の end-to-end reproducibility guard を完了した。exp073 は GPU train v2 `lgb_mean` CV 9.526374749、CPU deterministic train v1 `lgb_mean` CV 9.540138464、inference v1/v2 は byte-identical まで確認済み。raw `.csv.gz` は gzip metadata 差で揺れるため、feature determinism は decompressed content で見る。

- 2026-06-16 に `metric_weighted_tail_error_map` を実行した。exp027 anchor 版では exp027 / exp039 が visible train-derived sample をほぼ exact に再現し、非コピー候補では exp070 RMSE 4.341515、exp073 4.382939、exp063 4.533153、exp069 12.851383。exp073 anchor 版では exp070 が全体で exp073 より -0.041424 RMSE 良いが、改善は `00bbac68` と long-tail `1000+` に偏り、`000d7d20` / `00e12e8b` では exp063 が exp073 より良い。結論として、global blend / global replacement は支持せず、compact surface は exp075 の再現性担保済み生成物を使う long-tail geometry 条件付き候補、exp063 は short/mid bucket の比較候補、exp069 direct PF/Beam は棄却維持とする。
- 2026-06-18 に `exp078_compact_surface_longtail_gate` を実装し、exp073 base と exp075 compact surface OOF を align した。compact gate は global RMSE/SSE では改善し、最良 diagnostic は `tail_or_len_long_w020` で RMSE 9.526375 -> 9.362945、delta SSE -11,681,440 だった。ただし最大 well RMSE 悪化が 2.908365 で、long-tail / 評価指標 discussion 由来の worst-well guard 0.25 を大きく超えたため、submit candidate にはしない。今後 compact surface を再訪するなら、悪化 well を説明できる geometry / disagreement guard を先に作る。
- 2026-06-20 に `pf_beam_disagreement_error_map` を作成した。exp083 well summary / plot manifest と exp073 train_v2 by-well OOF metrics を join し、`studies/pf_beam_disagreement_error_map/` に overall / well / bucket map を保存した。全体では PF pooled RMSE 14.493061、Beam 15.774328、ML 9.526375 で、PF 直接置換は支持しない。一方で PF が ML に勝つ well は 234/773、best engine が PF の well は 207/773 ある。低 PF/Beam disagreement と小さい likPF delta は truth TVT に近づきやすく、high disagreement / high likPF delta / long tail は外れやすい。ただし high disagreement 内にも PF が大勝ちする well があるため、hard router ではなく `prefix_backtest_tvt_confidence`、`pf_beam_disagreement_sample_weight`、`pf_candidate_coverage_then_ranker_audit` の材料に集約する。
- 2026-06-20 に `exp086_oof_feature_importance_error_readout` の Kaggle train v1 を完了した。exp077 policy OOF predictions、exp077 fold-averaged feature importance、exp072 full replay feature cache を join し、exp073 baseline OOF error の feature bucket readout を作成した。baseline RMSE は 9.526374826、exp077 policy RMSE は 9.470514801。error lift は `pf_vs_dense`、`tvt_densew_d`、`tvt_dense50_d`、`tvt_dense_d`、`dense_dist`、`beam_std_d` に集中し、selected feature の absolute-error correlation は `beam_std_d` と `dense_dist` が強い。これは direct replacement ではなく、`pf_beam_disagreement_sample_weight` の confidence feature / sample weight 候補を絞る材料として扱う。
- 2026-06-20 に `exp087_prefix_backtest_tvt_confidence` の Kaggle train v2 を完了した。exp072 full replay cache 3,783,989 rows / 773 wells を使い、near-prefix calibration phase から PF/Beam confidence model を fold-safe に fit した。primary PF RMSE は 14.493050690、expected error vs absolute error Pearson は 0.519681049。confidence bin の observed MAE は low 2.460268 から high 19.057079 へ分離し、unstable flag high-error rate は 0.538377、stable は 0.115414。上位 signal は `pf_likpf_abs`、`md_since`、`pf_beam_abs`、`beam_likpf_abs`、`likpf_delta_abs`。`prefix_backtest_tvt_confidence` は完了済みとして backlog から外し、直接置換ではなく `pf_beam_disagreement_sample_weight` の feature / sample-weight 候補へ吸収する。
- 2026-06-20 に `exp089_pf_beam_disagreement_sample_weight` の Kaggle train v1 を完了した。exp073 control `lgb_mean` は RMSE 9.526374573 を再現し、sample-weight only の `sample_weight_unstable_downweight` が 9.521212047 で -0.005162526 改善した。一方で confidence feature add-only は 9.564240、feature+weight は 9.562019 で悪化し、PF/Beam disagreement feature をそのまま足す方向は支持しない。sample-weight only は global では小改善だが、well-level では improved 374 / worsened 399、最大悪化 +1.096752、距離 100-1000 bucket は悪化したため、submit 候補ではなく weight policy の小改善診断として閉じる。follow-up は PF/Beam confidence feature 追加ではなく、candidate ranker / physical likelihood / U-projection 完走に優先度を移す。
- 2026-06-20 に `exp091_self_gr_likelihood_pf_beam_probe` の Kaggle train v1 を完了した。exp072 full replay cache 3,783,989 rows / 773 wells で、既存 `pf_ancc` / `beam_mean` / `likpf_mean` / `sc_ens` / `hyb` と horizontal self-GR candidates を横並びに監査した。best single は `likpf_mean` RMSE 11.594897 / within10 0.772807。self-GR 単体は `self_gr_ens` RMSE 191.215912、`self_gr_best` RMSE 250.161697 と弱く、直接置換や hard switch は不採用。一方 oracle best は RMSE 6.873199 / within10 0.925153 / selected self-GR rate 0.135212 で headroom はあるが、現行 target-free `candidate_rank_score` top1 は RMSE 29.985529 と不十分。self-GR は候補値そのものではなく、後続 ranker feature 材料に限定する。
- 2026-06-20 に `exp093_pf_candidate_coverage_then_ranker_audit` の Kaggle train v1 を完了した。exp072 full replay cache 3,783,989 rows / 773 wells で、baseline candidate set (`pf_ancc` / `beam_mean` / `likpf_mean` / `sc_ens` / `hyb`) と baseline+self-GR を比較した。best single は `likpf_mean` RMSE 11.594897 / within10 0.772807。baseline oracle は RMSE 7.434030 / within10 0.906525、baseline+self-GR oracle は RMSE 6.958935 / within10 0.922492 で、候補集合には headroom がある。一方 target-free rank score top1 は baseline RMSE 12.507841、baseline+self-GR RMSE 29.985529 と弱い。候補別 selection count では `pf_ancc` が oracle best 1,092,069 rows なのに rank score top1 0 rows で、現行 scorer が PF ANCC を過小評価している。その後 exp101 で self-GR を外した5候補の supervised ranker / scorer audit を完了したが、best OOF は `likpf_mean` 単体を超えず不採用。候補集合 headroom は残るが、row-wise selector ではなく生成側・PF likelihood 側・continuity-constrained verifier 側で扱う。
- 2026-06-21 に `exp098_selector_rank_slot_features_on_exp073` の Kaggle train v1 と user-requested inference v1、2026-06-22 に user-submitted code submission を完了した。exp073/exp072 196-feature full replay surface に、target-free PF/Beam/likelihood-PF rank-slot structured features 64 個を追加し、合計 260 features で `rank_slot_u_disagreement` のみを学習した。pooled OOF は `lgb1` 9.358151052、`lgb2` 9.366698537、`lgb_mean` 9.427447987、`lgb0` 9.732275226。best `lgb1` は exp073 raw anchor 9.526374749 から -0.168223697、exp077 policy 9.470514801 から -0.112363749 改善したが、exp092 best `lgb1` 9.322479896 より +0.035671157 悪い。rank1 source distribution は `pf_ancc` 33.65%、`beam_mean` 24.55%、`likpf_mean` 41.80% で、`sc_ens` / `hyb` はほぼ選ばれなかった。inference は `lgb1` fold boosters 5 個で 14,151 rows、fallback 0、submit-check PASS、submission SHA `1d32582f3f5984eeb9dd0bc5798b12cdc2e7aa863e0334691028901f0325125f`。ユーザー訂正により `ref=53927479` Public LB 8.350 は exp092、nearby `ref=53927490` Public LB 8.441 は exp098 として記録する。exp098 8.441 は exp077 8.611 を -0.170 改善するため、rank-slot idea は有用と判断する。ただし exp092 8.350 には届かないため standalone anchor ではなく、compact / top-n rank-slot signals を exp092 に add-only merge する follow-up の材料として扱う。
- 2026-06-22 に `exp105_compact_rank_slot_features_on_exp098` の Kaggle train v1 を完了した。exp098 の 64 rank-slot features から pairwise delta、rank 間 U diff、`u_corr` / `u_resid` 符号反転ペア、fit degree、source flags を削り、22 compact features を base 196 features に足したが、pooled OOF は `lgb2` 9.441103161、`lgb1` 9.477699412、`lgb_mean` 9.506397523、`lgb0` 9.774440354 で悪化した。best `lgb2` は exp098 `lgb1` より +0.082952、exp098 `lgb_mean` より +0.013655、exp092 `lgb1` より +0.118623 悪い。compact 22-column set は rejected とし、提出しない。削った列のどれか、または full rank-slot の冗長性自体が LightGBM の分岐探索に効いていた可能性があるため、exp098 full rank-slot を比較基準として維持する。
- 2026-06-22 に `exp107_selector_topn_candidate_only_features` の Kaggle train v1 を完了した。exp098 と同じ exp073/exp072 196-feature surface に、rank slot に入った候補だけから作る top1/top2/top3 candidate-only features を追加し、3 variant x 3 LGBM x 5 folds の 45 boosters を学習した。best は `top2_candidate_only` / `lgb2` 9.437602823、`top2` / `lgb1` 9.437894828、`top2` / `lgb_mean` 9.479092683。best は exp073 raw 9.526374749 と exp077 9.470514801、exp105 best 9.441103161 は上回るが、exp098 `lgb1` 9.358151052 より +0.079451770、exp098 `lgb_mean` 9.427447987 より +0.010154835、exp092 `lgb1` 9.322479896 より +0.115122927 悪い。path continuity は best top2/lgb2 で step >=10 が 1、step >=25 が 0 で全体崩壊なし。追加 rank-slot 列だけを top-n candidate-only に削る方向は rejected、提出しない。exp098 full rank-slot を比較基準として維持する。
- 2026-06-22 に `exp108_topn_related_feature_prune` の Kaggle train v1 を完了した。exp098 の full 260 feature surface から top3 selector-related static column set だけを残し、195 features で学習した。pooled OOF は `lgb2` 9.479370656、`lgb1` 9.491034034、`lgb_mean` 9.529005954、`lgb0` 9.798771537。best `lgb2` は exp073 raw 9.526374749 より -0.047004 改善する一方、exp098 best 9.358151052 より +0.121220、exp105 best 9.441103161 より +0.038267、exp077 9.470514801 より +0.008856、exp092 best 9.322479896 より +0.156891 悪い。rank-slot U-shape features は残ったが、full surface の broad candidate/context/disagreement を削ると弱くなるため、topn-related static prune は rejected、提出しない。exp098 full rank-slot を比較基準として維持し、今後は prune より exp092 への小さな add-only merge または candidate-generation / likelihood 側を優先する。
- 2026-06-21 に `exp099_pf_multi_observation_likelihood_probe` の Kaggle train v1 を完了した。既存 `pf_ancc` / `beam_mean` / `likpf_mean` / `sc_ens` / `hyb` を raw horizontal GR の複数観測点 likelihood で target-free に再採点した。best single は既存 `likpf_mean` RMSE 11.594897 / within10 0.772807。baseline oracle は RMSE 7.434030 / within10 0.906525、baseline+multiobs oracle は RMSE 6.897510 / within10 0.922941 で、multiobs 追加により oracle RMSE は -0.536520、within10 は +0.016415 改善した。一方、target-free rank score top1 は `beam_mean` 偏重で RMSE 89.994392 / within10 0.523815 と崩壊し、multiobs 単体候補も最良 `likpf_multiobs_blend_w0p25` RMSE 25.110830 と弱い。multiobs は直接 scorer / direct replacement では不採用。multiobs score / MAE / NCC は exp101 の supervised ranker feature としても `likpf_mean` 単体を超えなかったため、直接候補 selector ではなく `learned_pf_observation_likelihood_probe` の PF observation likelihood 材料に限定する。

### 現行依存関係

古い依存関係の全量はこのファイルに残さない。完了済み実験の詳細、古い候補の失敗理由、日付つきログは `experiment_summary.md`、`SUBMISSIONS.md`、各実験の `SESSION_NOTES.md` / `result.md` を正とする。ここでは次の実験判断に直接効く依存だけを残す。

1. route別anchorはML Public-LB submitted anchorを`exp413` Public LB 7.201 (`ref=55080377`)、ML raw deterministic anchorを`exp073` Public LB 8.780、ensemble Public-LB referenceを`exp510` Public LB 7.201 (`ref=55231514`)とする。exp510はexp413と公開3桁同値、honest OOFなし、full-precision差不明のため、overall / ML anchorはexp413を維持する。旧ensemble referenceはexp494 7.228、exp082 7.601。直前ML anchorはexp335 7.517、exp287 7.530、exp264 7.562。exp413はreport-only tail悪化、exp335 / exp287 / exp264はtrain-side guard falseなので、LB referenceとtrain-side robust anchorを分離する。
2. exp148 / exp193 / exp198 / exp218 系 ML 比較では、CV だけでなく Public LB、near row、worst-well、exp115 hidden-like stress、raw-test parity を同時に見る。
3. exp205 / exp209 exact HMM 系は train-side では有望だが、raw-test-safe regeneration、runtime、worst-well regression が未解決。HMM 関連 backlog はこの guard を先に通す。
4. GRCAL-PFBEAM 系の exp211 / exp213 / exp214 / exp216 は direct generation では採用しない。top-K path、gap、dispersion、calibration quality などの confidence / selector feature 材料に限定する。
5. PF/Beam、GR、typewell、spatial prior は direct replacement / hard switch ではなく、selector、confidence feature、sample weight、diagnostic readout として扱う。
6. 公開 notebook replay は source-port で hidden rerun 可能な生成ロジックだけを候補にする。public output CSV copy や mount 不能 artifact 依存は採用しない。
7. stochastic PF/Beam / GPU 生成物は deterministic anchor にしない。feature content SHA、prediction SHA、submission SHA、kernel version が揃うまで比較候補扱いに留める.

### 共通参照元: `GRCAL-PFBEAM-20260707`

詳細はリンク先を正とし、この節では再現入口と運用ルールだけを残す。

- 調査メモ: [`public_discussion_notebook_catchup_2026-07-05.md`](../notebooks/rogii-wellbore-geology-prediction/public_discussion_notebook_catchup_2026-07-05.md)
- 公開notebook archive:
  - [`hujile/rogii-maybe-you-like`](../notebooks/rogii-wellbore-geology-prediction/date_run_recent_20260705/hujile__rogii-maybe-you-like/) / https://www.kaggle.com/code/hujile/rogii-maybe-you-like
  - [`lightningv08/rogii-lb-7-168`](../notebooks/rogii-wellbore-geology-prediction/date_run_recent_20260705/lightningv08__rogii-lb-7-168/) / https://www.kaggle.com/code/lightningv08/rogii-lb-7-168
  - [`yusuketogashi/rogii-lb7156-baseline`](../notebooks/rogii-wellbore-geology-prediction/date_run_recent_20260705/yusuketogashi__rogii-lb7156-baseline/) / https://www.kaggle.com/code/yusuketogashi/rogii-lb7156-baseline
  - [`pilkwang/working-note-target-free-tvt-geosteering`](../notebooks/rogii-wellbore-geology-prediction/date_run_recent_20260705/pilkwang__working-note-target-free-tvt-geosteering/) / https://www.kaggle.com/code/pilkwang/working-note-target-free-tvt-geosteering
  - [`georgymamarin/fork-the-ruler-not-the-model`](../notebooks/rogii-wellbore-geology-prediction/date_run_recent_20260705/georgymamarin__fork-the-ruler-not-the-model/) / https://www.kaggle.com/code/georgymamarin/fork-the-ruler-not-the-model
  - [`busyaprime/persistence-is-the-geosteering-baseline-to-beat`](../notebooks/rogii-wellbore-geology-prediction/date_run_recent_20260705/busyaprime__persistence-is-the-geosteering-baseline-to-beat/) / https://www.kaggle.com/code/busyaprime/persistence-is-the-geosteering-baseline-to-beat
- Discussions:
  - 716289 `Pointwise GR Makes No Sense`: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/716289
  - 717445 `FOYSAL writeup thread`: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/717445
- 再現入口: 各 archive directory の `kernel-metadata.json` と `code_file`。
- 再現ルール: Kaggle 上で再実行する場合は `competition_sources`、`dataset_sources`、`enable_gpu`、`machine_shape`、`enable_internet=false` を合わせ、実行ログ、`submission.csv` SHA、runtime、fallback / missing artifact を記録する。
- 実装ルール: ローカルでは notebook 丸ごと再実行より、必要セル・関数を抽出して同じ入力条件の study / exp として再実装する。Discussion 本文は Kaggle UI 参照範囲を `SESSION_NOTES.md` に明記する。
- 2026-07-07 に P0-A `affine_calibrated_gr_observation_pfbeam` は `exp211_affine_calibrated_gr_observation_pfbeam` として Kaggle train v1 を完了。478,958 rows / 64 wells、runtime summary 3,300.654 sec。affine fallback は 0/64 wells で、slope median 0.852239、prefix RMSE median 7.514946。Beam は `beam_affine_top1` が raw Beam RMSE 18.339188 -> 18.065010 と -0.274177 改善したが、best non-oracle は既存 `exp072_pf_ancc` RMSE 17.494197 のまま。PF/likPF は `pf_affine_lik_mean` RMSE 21.184758、`pf_affine_structural_lik_mean` 21.143708 で primary `pf_raw_lik_mean` 18.640063 から悪化した。したがって P0-A は direct PF/Beam generation 変更として完了/不採用とし、active backlog から外す。残す場合は affine slope/intercept、prefix RMSE、raw-vs-affine disagreement、oracle headroom を P2 `topk_path_confidence_features` などの selector/confidence feature 材料に限定する。
- 2026-07-07 に P0-B `prefix_structural_prior_pfbeam` は `exp213_prefix_structural_prior_pfbeam` として Kaggle train v1 を完了。478,958 rows / 64 wells、runtime summary 3,415.409 sec。raw GR観測は固定し、known prefix の `TVT_input + Z` surface fit から PF 初期速度、PF velocity pull、Beam absolute / step-delta soft cost、Beam top-K path diagnostics を生成した。Beam は `beam_structural_base_top1` が raw Beam RMSE 18.339188 -> 18.312677 と -0.026510 小改善し、distance bucket 全体で小幅改善したが、best non-oracle は既存 `exp072_pf_ancc` RMSE 17.494197 のまま。PF は `pf_structural_weak_lik_mean` 28.230909、`pf_structural_base_lik_mean` 29.564037、`pf_structural_slope_only_lik_mean` 30.621856 で primary `pf_raw_lik_mean` 21.081279 から大きく悪化した。したがって P0-B は direct PF/Beam generation 変更として完了/不採用とし、active backlog から外す。P0-A/P0-B の単独結果が弱いため P0-C `calibrated_gr_plus_prefix_structural_prior_pfbeam` の direct generation follow-up も進めない。残す場合は Beam top-K gap、path spread、raw-vs-structural disagreement、affine/prefix calibration quality を P2 `topk_path_confidence_features` などの selector/confidence feature 材料に限定する。
- 2026-07-07 に P1 `public_raw_gr_residual_scale_control` は `exp214_public_raw_gr_residual_scale_control` として Kaggle train v1 を完了。478,958 rows / 64 wells、runtime summary 3,369.568 sec。public-like `TVT + Z` surface-state likelihood-PF、500 particles x 128 seeds、scale 3/5/8/12 で raw GR + known-prefix residual scale control を固定した。primary `pf_raw_scale_5` は RMSE 15.596465、best non-oracle `pf_raw_scale_12` は RMSE 15.223857、oracle best は RMSE 11.104328、`pf_raw_top3_oracle` は RMSE 14.236926。reference は `exp072_pf_ancc` RMSE 17.494197、`beam_raw_top1` 18.339188、`exp072_pf_z` 24.165177。これは exp211/213 の軽量 raw controls より強く、P0-A/P0-B 比較には public-like control が必要だったことを確認した。direct inference / submit はしない。今後は exp214 `pf_raw_scale_*` を GRCAL-PFBEAM 診断の固定 raw control として使い、scale/top-seed/oracle headroom は P2 `topk_path_confidence_features` の材料に限定する。
- 2026-07-07 に P1 diagnostic `affine_shift_landscape_ruler_readout` は `exp216_affine_shift_landscape_ruler_readout` として Kaggle train v1 を完了。3,561,984 row-context rows / 773 wells、runtime 1264.923 sec。best overall は `savgol_31_p2__raw` RMSE 108.534313、hidden_tail best は `rolling_median_11__raw` RMSE 125.707127。raw smoothing は surface sharpness と prefix_backtest で小改善したが、affine/heel calibration は hidden_tail mean abs-error gain が `raw__heel_calibrated` -2.056464、`rolling_median_11__heel_calibrated` -2.075805、`savgol_31_p2__heel_calibrated` -2.783559 と悪化し、`likpf_mean` observation rank も raw より悪化した。したがって active backlog から外し、direct generation / candidate replacement / inference / submit はしない。残す場合は `zero_rank`、entropy、secondary-mode / bimodal signal、calibration residual scale を P2 `topk_path_confidence_features` の uncertainty / fallback 特徴量候補に限定する。
- 2026-07-08 に `exp072_exp205_joint_exact_parity_fast_cache_generation` は `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation` Kaggle train v5 を完了。`outer_workers=2`, `numba_num_threads=2` により total runtime は 20,203.290 sec (約 5h36m43s) となり、v4 から 12,580.970 sec 短縮、exp072 v2 + exp205 v2 の単純合算 32,770.755 sec より 12,567.465 sec 速い。HMM elapsed は 11,285.868 sec で v4 から 8,463.231 sec 短縮。HMM decompressed feature SHA は exp205 v2 と一致し、best candidate も `blend_likpf_hmm_w500` で一致。best RMSE は 10.269696147 で exp205 v2 との差 3.8106e-06 は、ユーザー確認により近似一致として許容する。一方、exp072 full replay cache は raw gzip SHA / decompressed SHA とも exp072 v2 reference に不一致で、full artifact exact parity は未証明。2026-07-09 に v6 `outer_workers=4`, `numba_num_threads=1` も完了したが、total 28,768.406 sec、HMM 14,627.100 sec で v5 よりそれぞれ 8,565.116 sec / 3,341.232 sec 遅かった。HMM SHA と RMSE 近似一致は維持したが不採用とし、採用設定は v5 の `outer_workers=2`, `numba_num_threads=2` のままにする。近似 RMSE 基準では runtime target 達成として完了扱いにする。
- 2026-07-08 に `gr_wavelet_rotation_confidence_features_on_exp148` は `exp218_gr_wavelet_rotation_confidence_features_on_exp148` として Kaggle train v1 / inference v1 / submission を完了。train は 3,783,989 rows / 773 wells / 380 features / 15 boosters、GRWR generated features 86、feature join coverage pass。pooled OOF は `lgb0` 8.557165712、`lgb1` 8.512227651、`lgb2` 8.524447601、`lgb_mean` 8.475793752。exp148 GPU `lgb_mean` 8.501281182 から -0.025487430 改善したが、exp160 `lgb_mean` 8.463718774 より +0.012074978 弱い。1000+ tail は -0.030207 改善し、near 000_050 も -0.021092 改善した一方、100-1000 bucket は小幅悪化。by-well は 413 wells 改善 / 360 wells 悪化、最大悪化 `f88ddb26` +4.075520。`grwr_fft_rotation_ratio_x_log1p_md_since` は feature importance 全体 4 位で GRWR block は効いている。inference v1 は current-test GRWR replay と saved `lgb_mean` 15 boosters で 14,151 rows、fallback 0、prediction SHA `483845c8969e99e8d12c9dfcbe43bb8dfc727a1df8905ef045f02e35ebdcbff1`、submission SHA `77a2c2804749dc811ba61f43d9d8827c69282e83e116233559da80b6820c0824`、submit-check PASS。submission ref `54457577` は Public LB 7.843 で、exp148 CPU runtime 7.921 から -0.078、exp148 GPU 7.960 から -0.117、exp198 7.930 から -0.087 改善したため、ML route submitted anchor を exp218 に更新する。overall は exp082 ensemble 7.601 が引き続き最良。backlog は完了として外す。
- 2026-07-08 に `ml_tvt_typewell_gr_mismatch_error_detector_on_exp148` は `exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148` として Kaggle CPU train v1 を完了。3,783,989 rows / 773 wells / 35 feature columns。exp148 base RMSE は 8.501281182。primary `mlgr_mismatch_signal` は `abs_error_gt10` AUC 0.573943 で採用目安 0.65 に届かず、q90 high-mismatch bucket は error_gt_rate 0.234520 / error_gt_lift 1.632373、abs_error_lift 1.425989 と risk bucket は捉えるが単独 detector として弱い。diagnostic correction は base exp148 が最良で、`best_offset` 補正は採用しない。したがって active backlog から外し、exp148/exp193 add-only LightGBM、inference、submit には進めない。残す場合は weak risk flag / bucket readout として将来の confidence ensemble 材料に限定する。
- 2026-07-08 に `row_neighbor_input_context_features_on_exp148` は `exp220_row_neighbor_input_context_features_on_exp148` として Kaggle CPU split train v1 を完了。`train_lgb0` / `train_lgb1` / `train_lgb2` はすべて `COMPLETE`、各 1 LightGBM config x 5 folds = 5 boosters、合計 15 boosters、control / parent 再学習なし。3 split OOF を streaming aggregate した `lgb_mean` は 8.496282588。exp148 GPU historical `lgb_mean` 8.501281182 からは -0.004998594 改善したが、exp193 8.456665439、exp198 8.457923653、現行 ML route submitted anchor exp218 8.475793752 には届かないため、inference / submit はしない。`rnic_` importance は `likpf_mean_d` と `uproj_source_u_std` の lead/lag 差が上位で、将来使う場合は row-neighbor block 単体ではなく exp218 以降の feature set への小さな補助として扱う。backlog は完了/不採用として外す。
- 2026-07-09 に `joint_typewell_self_gr_hmm_likelihood_probe` は `exp223_joint_typewell_self_gr_hmm_likelihood_probe` として Kaggle train v1 を完了。CPU-only / 2 HMM variants / 0 boosters、3,783,989 rows / 773 wells、elapsed 39,029.366 sec (約10h50m29s)。best は `hmm_selfgr_boost_only_a070_c100` で RMSE 11.349950650、exp072 `likpf_mean` 11.594897668 から -0.244947018、MAE -0.596360991、within10 +0.022027812。distance bucket は全 bucket で改善し、hidden-like も verification_like_spatial -1.180418、typewell_purged -1.240497 と改善した。一方、exp209 HMM/likPF blend RMSE 10.269696 には届かず、by-well は 461 improved / 312 worsened、最大悪化 `b19b0395` +46.954683 RMSE と大きい。したがって self-GR HMM weak boost は train-side signal としては支持するが、raw-test regeneration / inference / submit には進めない。後続で使う場合は直接候補や replacement ではなく、ML / selector 側の confidence feature または regression guard 付き診断材料に限定する。backlog は完了として外す。
- 2026-07-10 に `state_known_tvt_self_gr_hmm_emission` は `exp225_state_known_tvt_self_gr_hmm_emission` として Kaggle train v1 を完了。CPU-only / 1 HMM variant / 0 boosters、3,783,989 rows / 773 wells、elapsed 17,310.949 sec (約4h48m31s)。candidate state が known-prefix TVT 範囲内の場合だけ self-GR `TVT_input -> GR` curve boost を足す実装は成立したが、`hmm_selfgr_state_known_tvt_curve_boost_only_a070_c100` は RMSE 14.212954500 で exp072 `likpf_mean` 11.594897668 から +2.618056832 悪化した。近傍 bucket は `000_050` -0.235925、`050_100` -0.316982 と改善したが、`1000_plus` は +2.931795、hidden-like は spatial +2.937794 / typewell-purged +2.842109 と悪化。by-well は 379 improved / 394 worsened、最大悪化 `2fd68f7b` +49.423573 RMSE。したがって state-known self-GR emission は完了・不採用とし、追加 grid、raw-test regeneration、inference、submit は行わない。self-GR を HMM emission に直接足す方向は下げ、使う場合は ML / selector confidence feature または regression guard readout に限定する。
- 2026-07-11 に `dtw_typewell_warp_hmm_emission_correction` は `exp230_dtw_typewell_warp_hmm_emission_correction` として Kaggle train v2 を完了。親はユーザー指定どおり `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`。CPU-only / 2 DTW-HMM variants / 0 boosters、3,783,989 rows / 773 wells、elapsed 36,768.8 sec (約10h12m49s)。best DTW-HMM は `hmm_dtw_a005_s1200` RMSE 13.611292323 で exp072 `likpf_mean` 11.594897668 から +2.016394654 悪化し、`hmm_dtw_a010_s1200` は 16.435494713 まで悪化した。near buckets は `000_050` -0.279896、`050_100` -0.424749、`100_250` -0.224815 と小改善したが、`1000_plus` +2.300709、hidden-like spatial +1.645723 / typewell-purged +1.740356、worst regression `b19b0395` +47.803293 が大きい。したがって DTW/typewell warp を HMM emission に直接足す方向は完了・不採用とし、raw-test regeneration、inference、submit、追加 alpha grid は行わない。使う場合は ML / selector 側の confidence feature または regression guard readout に限定する。
- 2026-07-12 に `exact_hmm_posterior_bimodality_audit` は `exp236_exact_hmm_posterior_bimodality_audit` として Kaggle CPU train v1 を完了。exp221 の固定 Gaussian-emission exact HMMを 3,783,989 rows / 773 wells で再生し、posterior mean は RMSE 8.327728486（親記録との差 -0.000008465）だった。二峰 row は 35,399（0.9355%）/ 138 wells / 317 segments、mean-in-valley row は 6,781（0.1792%）、mode mass switch は17回で、頻繁な mode slip は確認されなかった。marginal MAP は RMSE 8.365160435（+0.037431949）、dominant-mode conditional mean は 8.331754352（+0.004025866）で全体悪化し、MAP の `|step delta| > 0.2` は 3.2659% と posterior mean の 0.0219% より大きかった。oracle top2 は二峰 rowに未使用の選択 headroom を示すだけである。したがって posterior decoder の直接変更、mixture emission、mode-state、raw-test regeneration、inference、submitは行わない。二峰/谷/entropy signalを残す場合も、raw-test parityとfold-safe生成を満たす add-only confidence 特徴量に限定する。

### 完了済み詳細メモ

- 2026-07-02 に `exp165_coordinate_frame_normalization_features_on_exp148` の Kaggle CPU split train と Kaggle 内 aggregate を完了した。3,783,989 rows / 773 wells / 331 features / 15 boosters、pooled OOF は `lgb0` 8.623039477、`lgb1` 8.586673413、`lgb2` 8.616753590、3-model `lgb_mean` 8.549931602。exp148 historical `lgb_mean` 8.501281182 から +0.048650420 悪化したため、coordinate-frame normalization add-only は完了/不採用として backlog から外し、inference port / submit はしない。座標系は direct TVT candidate、hard correction、row-wise selector には展開しない。
- 2026-07-03 に `exp166_prefix_crop_midwindow_replacement_on_exp148` の Kaggle CPU 2段階 cache/train を完了した。feature cache は 3,783,989 rows / 773 wells / 96 features、split train は `lgb0/lgb1/lgb2` x `tail500/tail1000` replacement-only を完了。best single は `tail500/lgb0` CV 8.566426970、`tail1000` best は `lgb2` CV 8.574216683。exp148 `lgb_mean` 8.501281182 からそれぞれ +0.065145788 / +0.072935501 悪化し、exp161 last50 add-only best single 8.564724996 にも届かなかった。したがって mid-window replacement-only は完了/不採用として、inference port / submit はしない。案1 last50 replacement-only は isolated test として残るが、prefix crop-window 系の優先度は下げる。
- 2026-07-04 に `exp172_prefix_crop_last50_replacement_only_on_exp148` の Kaggle CPU 2段階 cache/train を完了した。feature cache は 3,783,989 rows / 773 wells / 48 features、split train は `prefix_crop_last50_multiobs_replacement` の `lgb0/lgb1/lgb2` 15 boosters を完了。pooled OOF は `lgb0` 8.583559279、`lgb1` 8.575126850、`lgb2` 8.586986606。best `lgb1` でも exp148 `lgb_mean` 8.501281182 から +0.073845668 悪化し、exp161 last50 add-only best single 8.564724996、exp166 tail500 replacement-only best single 8.566426970 にも届かなかった。したがって last50 multiobs replacement-only は完了/不採用として backlog から外し、inference port / submit はしない。prefix crop-window 系は add-only / replacement-only では閉じ、残すなら source prefix を先に last50 へ切る feature-rebuild 仮説だけを低優先で扱う。
- 2026-07-02 に `exp167_fft_denoised_gr_matching_audit` の Kaggle train v2 を完了した。raw train 773 wells、395,776 sampled rows で raw / rolling median / Savitzky-Golay fallback / FFT notch の typewell GR shift-scan surface を比較した。FFT notch は raw に対して all RMSE 108.659395 -> 108.581079、hidden_tail RMSE 125.711348 -> 125.580817 と小幅改善したが、hidden_tail gap gain -0.003365、decoy gap gain -0.000498 で surface の鋭さは改善しなかった。prefix_backtest も MAE / within2 / within5 が悪化した。一方 rolling median / Savitzky-Golay は gap、entropy、decoy gap を改善した。したがって `fft_denoised_gr_matching_audit` は完了/不採用として backlog から外し、FFT notch を `denoised_gr_pfbeam_generation_audit` へ直接進めない。続ける場合は FFT ではなく rolling/savgol smoothing 単体の小さな observation audit として扱う。
- 2026-07-02 に `exp170_heel_calibrated_shift_scan_pfbeam_audit` の Kaggle train v1 を完了した。raw train 773 wells、hidden_tail 197,888 rows / prefix_backtest 197,888 rows で raw / flat-calibrated / heel-calibrated typewell GR surface と fixed exp072 PF/Beam candidate observation cost を比較した。heel calibration は raw に対して hidden_tail mean abs-error gain `raw__heel_calibrated` -2.058258、`rolling_median_11__heel_calibrated` -2.076816、`savgol_31_p2__heel_calibrated` -2.784873 と悪化し、flat calibration は -27ft 級で大きく壊れた。PF/Beam observation も `likpf_mean` の mean gap だけは 10.852016 -> 10.771421 と小さく下がったが、mean rank 18.163254 -> 19.346994、top1 rate 0.052105 -> 0.045445、top5 rate 0.233162 -> 0.215344 と悪化した。したがって `heel_calibrated_shift_scan_pfbeam_audit` は完了/不採用として backlog から外し、heel calibration を PF/Beam likelihood / initial offset prior / exp148 ML feature へ進めない。GR alignment を続けるなら、exp167 の rolling/savgol smoothing を小さく見るか、bimodal posterior / mode ambiguity diagnostics を優先する。
- 2026-07-04 に `exp189_denoised_gr_pfbeam_generation_audit` の Kaggle train v1 を完了した。exp072/099 と同じ `TVT_input_missing_equivalent_exp063_rows` で 64 wells / 478,958 rows を対象に、raw / rolling median w11 / Savitzky-Golay w31 p2 の GR observation likelihood を同一 PF seed / particles / Beam 幅で比較した。primary `pf_raw_lik_mean` は RMSE 20.225464、best generated non-oracle `beam_rolling_median_w11_top1` は RMSE 18.028587 で raw Beam 18.339188 から -0.310600 改善したが、reference `exp072_pf_ancc` RMSE 17.494197 には届かず、max well regression も +17.732656。`beam_savgol_w31_p2_top1` も RMSE 18.136752 で同様に小幅改善止まり。PF likelihood smoothing は `pf_rolling_median_w11_lik_mean` RMSE 26.893376、`pf_savgol_w31_p2_lik_mean` RMSE 27.943343 と raw PF から大きく悪化した。ESS は raw 175.913 -> rolling 177.722 / savgol 180.394、resampling rate は 0.0512 -> 0.0393 へ下がるが、候補 TVT としては wrong depth へ安定して吸い込まれる可能性が高い。一方 oracle best smoothed は RMSE 10.643257 で headroom はあるため、残す場合は `denoised_calibrated_matching_features_on_exp148` の selector / ML confidence feature 材料に限定する。`denoised_gr_pfbeam_generation_audit` backlog は完了/診断のみとして外し、PF/Beam generation likelihood の直接変更、direct replacement、inference port、submit はしない。
- 2026-07-04 に `exp190_denoised_calibrated_matching_features_on_exp148` の Kaggle train v1 を完了した。exp148 learned-likelihood ML anchor に raw / rolling median / Savitzky-Golay の GR shift-scan sharpness、posterior ambiguity、candidate disagreement、prefix backtest quality を add-only で追加し、3,783,989 rows / 773 wells / 431 features / 15 boosters を学習した。pooled OOF は `lgb0` 8.601678275、`lgb1` 8.539624480、`lgb2` 8.540073562、`lgb_mean` 8.503596159。`lgb1` 単体は exp148 同 config から -0.024346641 改善したが、採用基準の `lgb_mean` は exp148 `lgb_mean` 8.501281182 から +0.002314978 悪化した。したがって `denoised_calibrated_matching_features_on_exp148` backlog は完了/不採用として外し、current-test parity 実装、inference port、submit はしない。GR matching / posterior signal を続ける場合は、今回の一括 DCM block ではなく、exp157/158/183 系 selector confidence や feature importance で支持された小さい subset に限定する。
- 2026-07-02 に `exp171_bimodal_posterior_pfbeam_candidate_audit` の Kaggle train v1 を完了した。raw train 773 wells、1,187,328 row-context rows で target-free GR shift-scan top2 local minima から固定温度 posterior mean 候補を作り、hard commit / midpoint / fixed exp072 PF/Beam candidates と比較した。best fixed candidate は `likpf_mean` RMSE 11.471434 / MAE 6.989252 / within10 0.775439。best posterior all は `rolling_median_11/posterior_mean_t16` RMSE 76.698097、hidden_tail best posterior は `savgol_31_p2/posterior_mean_t16` RMSE 102.301054。posterior / midpoint は hard commit より mean abs-error を最大 +1.328095ft 改善したが、`likpf_mean` には大きく届かない。したがって `bimodal_posterior_pfbeam_candidate_audit` は完了/不採用として backlog から外し、posterior candidate direct replacement、PF/Beam likelihood 変更、inference port、submit はしない。`p`、entropy、mode separation、top2 gap も現状実装では exp148 add-only feature へ進める根拠が弱い。
- 2026-07-03 に `exp169_tvt_input_pfbeam_offset_calibration` の Kaggle train v1 を完了した。known prefix 末尾 256 rows を holdout として PF/Beam replay し、773 wells / 197,888 prefix rows から candidate 別 offset を推定して exp072 fixed tail candidate へ capped/fade-in 補正を監査した。baseline `likpf_mean` は RMSE 11.594897672 / MAE 7.067632584 / within10 0.772807479。best offset correction `off_likpf_mean_self_median_a0p5_c10_g50_f250_iqr20_n32_const` は RMSE 11.580455166 で -0.014442507 改善したが、MAE 7.097507839、within10 0.772440935 と悪化し、max well regression +4.173820317 が残った。prefix offset 自体は安定していたが tail への direct correction は guard を満たさない。したがって `tvt_input_pfbeam_offset_calibration` は完了として backlog から外し、direct correction / inference port / submit はしない。残す場合は exp148 系 ML confidence feature に限定し、低優先で扱う。
- 2026-07-03 に `exp173_beam_topk_path_posterior_audit` の Kaggle train v2 を完了した。exp072 fixed train pseudo-tail cache 3,783,989 rows / 773 wells に対し、Beam search 本体を再実行して retained top-K path/cost、top1/top2 commit、top-K weighted mean、固定温度 posterior mean、entropy、path separation、top-K oracle headroom を保存した。primary baseline `likpf_mean` は RMSE 11.594897672。best posterior `beam_topk_sm11_bw64_posterior_mean_t16` は RMSE 15.972927962 / MAE 10.852413073 / within10 0.602145249 で `likpf_mean` から +4.378030290 悪化し、best top-K oracle `beam_topk_sm11_bw64_topk_oracle` でも RMSE 15.549454381、delta +3.954556709 と届かなかった。したがって `beam_topk_path_posterior_audit` は完了/不採用として backlog から外し、top-K posterior direct replacement、PF/Beam likelihood 変更、confidence feature 化、inference port、submit はしない。Beam top-K 系の後続案は独立 backlog 化せず閉じる。
- 2026-07-03 に `exp177_beam_topk_bimodal_gate_posthoc_audit` の Kaggle train v1 を完了した。exp173 保存済み top-K diagnostics / candidate_wide を読み、`likpf_mean` を default としたまま、二峰性 / low-cost-gap / entropy / spread gate 成立 row だけ `posterior_mean_t*`、`top2_commit`、`topk_weighted_mean` へ置換する no-training audit を行った。baseline `likpf_mean` は RMSE 11.594897884。最良 policy `beam_topk_sm11_bw64__and_sep_ge_q90__cost_le_q10__replace_posterior_mean_t1` でも RMSE 11.837783911 で +0.242886027 悪化し、changed subset は 10.269740849 -> 12.706185740、max well regression +22.519192863。near、1000+ longtail、Beam-likPF gap top quartile もすべて悪化した。したがって `beam_topk_bimodal_gate_posthoc_audit` は完了/不採用として backlog から外し、Beam top-K posterior / top2 / weighted mean の direct replacement、confidence feature 化、inference port、submit、大きな Beam top-K 再生成 follow-up には進めない。
- 2026-07-03 に `exp174_typewell_late_range_ml_posthoc_clip_audit` の Kaggle train v1 を完了した。exp148 `lgb_mean` OOF 3,783,989 rows / 773 wells に対し、known_last_pct が高い well で ML 予測の pred_pct が typewell TVT 前半へ落ちる row だけ shrink / clip する no-training grid を監査した。baseline は RMSE 8.501281182。lower bound `0.55/0.60/0.65` は changed_rows 0 の no-op、発火する best policy `fixed_lb0p7_klp0p75_a0p25` でも changed_rows 2,098 / 2 wells、RMSE 8.501891 で +0.000609 悪化した。最大発火 `known_last_m0p05_klp0p75_a0p25` は changed_rows 13,657 / 14 wells、RMSE 8.518425 で +0.017144 悪化。したがって `typewell_late_range_ml_posthoc_clip_audit` は完了/不採用として backlog から外し、ML hard lower-bound posthoc、inference port、submit はしない。late-range prior を続ける場合も PF/Beam candidate feature / selector prior に限定し、hard invalid / direct clip は避ける。
- 2026-07-04 に `exp176_typewell_late_range_pfbeam_candidate_prior` の Kaggle train v3 を完了した。exp157 supervised candidate ranker に target-free typewell late-range prior feature を追加し、candidate-long memory fix として row-level `tlp_` 複製を避けた。best OOF `lgb_candidate_error_ranker` は RMSE 10.641298 / MAE 6.434563 / within10 0.791815 / oracle acc 0.257439。`likpf_mean` 11.594897672 から -0.953600、exp157 best OOF 10.795800 から -0.154502、exp158 best Viterbi 10.789163 から -0.147865 改善した。一方 max path switch は 330.842 / 1000 rows と高く、row-wise direct selector としては不安定。したがって `typewell_late_range_pfbeam_candidate_prior` は完了/支持として backlog から外し、direct submit、hard invalid、clip、PF/Beam generation soft prior には進めない。次は exp158-style continuity selector または exp148/ML anchor confidence feature に限定する。
- 2026-07-04 に `exp186_typewell_late_range_pfbeam_generation_soft_prior` の Kaggle train v3 を完了した。v1/v2 の 192-row prefix-holdout audit は意図した full replay cache rebuild ではなかったため superseded とし、v3 で raw train horizontal/typewell から exp072-style full replay train feature cache を作り直した。既存 full replay cache は generation input として使っていない。出力は 3,783,989 rows / 773 wells / 196 features、runtime は summary 15,783.764 sec、feature generation 14,053.477 sec。selected soft prior は `pct50_strong2_pct70_weak0p5` で、PF_ANCC、PF_Z、Beam、128-seed likelihood-PF に適用した。train feature raw gzip SHA は `4bb7a43278ec65143d61c3451353735093995d5258aad665b901237a6a469185`、decompressed SHA は `b4dd75312d91b21f55b8d1ad09a8590c6bb75857ddfbbbc84d7db175dbb75d15`。exp072 full replay cache との direct PF/Beam RMSE TVT 比較では、`pf_ancc` が 14.493061 -> 14.220030、`pf_z` が 17.788174 -> 17.679589、`beam_mean` が 15.774328 -> 15.753703 と小改善したが、最強候補 `likpf_mean` が 11.594898 -> 12.942278 へ +1.347381 悪化した。したがって exp072 replacement としては不採用で、model training / inference / submit には進めない。soft prior を続ける場合は generation cache 全体ではなく、`pf_ancc` / Beam mean の局所改善を selector feature や candidate scoring の材料として切り出す。
- 2026-07-04 に `typewell_late_range_continuity_selector_on_exp176` を `exp191_typewell_late_range_continuity_selector_on_exp176` として実装し、Kaggle train v1 `kentookumura/exp191-typewell-late-continuity-train` を完了した。exp176 v3 saved boosters 15 本と exp176 feature schema を読み、exp176 v3 と同じ `tlp_` / `candidate_tlp_` feature contract で OOF score surface を復元し、exp158-style Viterbi 180 variants を評価した。best は `viterbi_sw400_bias000_jw050_jf025_d075_std999999_md0000_seg012` で RMSE 10.598006880 / MAE 6.402336928 / within10 0.793110657。`likpf_mean` 11.594897672 から -0.996890792、exp176 row-wise 10.641296371 から -0.043289491、exp158 best Viterbi 10.789163253 から -0.191156373 改善した。path switches は exp176 row-wise の 261,391 / 69.078 per 1000 rows から 3,620 / 0.957 per 1000 rows まで低下した。一方 near / mid distance bucket は小幅悪化し、356 wells は exp176 row-wise から悪化したため、selected TVT の direct inference / submit はしない。`typewell_late_range_continuity_selector_on_exp176` backlog は完了として外し、後続で使う場合は `exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148` として exp148 の既存 `learned_likelihood_confidence` (`ll_*`) block を置き換える実験に限定する。
- 2026-07-05 に `exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148` の Kaggle CPU split train v1 を完了した。exp148 の `learned_likelihood_confidence` (`ll_*`) block を外し、exp191 continuity selector の predicted-error / selected-family / typewell pct / segment-stability features に置き換える replacement-only ML 実験。CPU 実行の timeout 対策として train は `lgb0` / `lgb1` / `lgb2` の 3 notebook に分割し、各 1 config x 5 folds、合計 15 boosters、parent/control 再学習なし。pooled OOF は `lgb0` 9.464292702、`lgb1` 9.331742862、`lgb2` 9.313152706、3 split 平均 `lgb_mean_split3` 9.321908826。exp148 `lgb_mean` 8.501281182 から +0.820627644、exp193 `lgb_mean` 8.456665439 から +0.865243388 悪化した。exp194 replacement-only `lgb_mean` 9.329893102 よりは -0.007984276 良いが、`ll_*` block の代替としては不十分なため、current-test feature generation / inference port / submit は行わない。`exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148` backlog は完了/不採用として外す。
- 2026-07-04 に `exp192_typewell_late_range_hard_window_pct50_full_cache_replacement` の Kaggle train v1 を完了した。exp186 corrected full replay 実装を親に、soft prior を無効化し、raw typewell 読み込み直後に元 TVT range の `typewell_pct >= 0.50` filter を入れて PF_ANCC / PF_Z / Beam / 128-seed likelihood-PF を再生成した。出力は 3,783,989 rows / 773 wells / 196 features、runtime は 13,275.591 sec、raw gzip SHA は `1040d7d3b9254b5a36d2a3f7fd526ae28e3ddd5b29059926b44bbe9d84436e6a`、decompressed SHA は `a86dff450b108e4481208a5f5699f8624eaf736cb6eb6aa735d39b4044c6f0e1`。exp072 direct PF/Beam 比較では `pf_ancc` が 14.493061 -> 13.821178、`beam_mean` が 15.774328 -> 15.677016、`likpf_mean` が 11.594898 -> 11.544812 と改善した。一方 `pf_z` は 17.788174 -> 19.705112 と悪化し、true typewell pct `<0.50` subset では `likpf_mean` が +25.754210 RMSE、`beam_mean` が +32.502607 RMSE 悪化した。したがって direct cache candidate としては支持するが、PF/Beam route の direct inference / submit には進めない。続ける場合は downstream ML replacement-only で `pf_z` 悪化と early-range exception を吸収できるか確認する。`typewell_late_range_hard_window_pct50_full_cache_replacement` backlog は完了として外す。
- 2026-07-05 に `typewell_late_range_hard_window_pct40_full_cache_replacement` / `exp196_typewell_late_range_hard_window_pct40_full_cache_replacement` の Kaggle train v1 と direct PF/Beam comparison を完了した。3,783,989 rows / 773 wells / 196 features、runtime 8,616.007 sec、raw gzip SHA `7b1f51b1c4de16bbff59c9a0c1bd015fc3b6d6152c32a3a93b2f5a694a37576b`。exp072 比では `likpf_mean` 11.594898 -> 11.576062、`pf_ancc` 14.493061 -> 14.020904、`beam_mean` 15.774328 -> 15.711042 と改善したが、`pf_z` は 17.788174 -> 18.834133 と悪化。exp192 pct50 比では `pf_z` が -0.870979 RMSE、true typewell pct `<0.50` bucket も大幅改善した一方、global `likpf_mean` は +0.031251、`pf_ancc` は +0.199726 と悪化した。pct40 は pct50 の early-range exception と `pf_z` regression を緩める感度実験として支持するが、direct PF/Beam submit はしない。次に使う場合は downstream ML replacement-only で pct40 と pct50 を同条件比較する。
- 2026-07-05 に `pf_step_delta_soft_prior_full_replay_replacement` / `exp200_pf_step_delta_soft_prior_full_replay_replacement` の Kaggle train v1 と exp072 direct comparison v5 を完了した。raw train horizontal/typewell から 3,783,989 rows / 773 wells / 196 features の full replay cache を再生成し、`id` mismatch 0。`delta_free010_cost0025_scale003` step-delta prior は `likpf_mean` の near/mid bucket を大きく改善したが、overall RMSE は 11.594898 -> 11.618341 で +0.023444 悪化し、許容 +0.02 guard を超えた。`pf_ancc` も 14.493061 -> 14.736794 と悪化したため、direct replacement、inference、submit、追加 grid には進めない。backlog は完了/不採用として外し、残す場合も short-distance confidence diagnostics の材料に限定する。
- 2026-07-05 に `exp193_typewell_late_interval_context_features_addonly_on_exp148` の Kaggle train v1 `kentookumura/exp193-typewell-late-context-exp148-train` を完了した。exp148 ML anchor を親に、raw typewell TVT range と observed `TVT_input` prefix から `tlic_` context 19 features を add-only で足した。candidate_pct / candidate別 violation / exp176 selected TVT / direct clip / blend / postprocess は入れていない。3,783,989 rows / 773 wells / 313 features / 15 boosters、parent/control 再学習なし。pooled OOF は `lgb0` 8.553543817、`lgb1` 8.475340902、`lgb2` 8.510015021、`lgb_mean` 8.456665439。exp148 GPU `lgb_mean` 8.501281182 から -0.044615743 改善し、exp160 CV 8.463718774 もわずかに上回った。feature importance では `tlic_known_last_pct` rank 46 / 313、late-delta 系も rank 89/109/114 に入り、context-only signal は使われている。train-side supported とし、元の `typewell_late_interval_context_features_addonly_on_exp148` backlog は完了扱いにする。同日に same-exp inference v2 `kentookumura/exp193-typewell-late-context-exp148-inference` も完了。v1 は `generator.candidates` 欠落で失敗し、exp145/exp148 と同じ generator block 追加後の v2 は 14,151 rows、313 features、`tlic_` 19 features、fallback 0、train manifest/schema exact match、submit-check PASS。submission SHA256 は `9265e3e19e7eea20c6e0097b3b581b4a15c29353ebb77875d09ac30475502695`。code submission ref `54347471` は Public LB 7.946 で完了し、exp148 GPU inference v7 Public LB 7.960 からは -0.014 改善した。一方、ユーザー確認済みの exp148 CPU runtime inference Public LB 7.921 (`ref=54183122`) には +0.025 届かないため、exp193 は ML route submitted anchor には採用しない。CV 改善量より LB 改善量は小さく、CV-to-LB 転移は控えめ。アンサンブル route anchor の exp082 Public LB 7.601 は引き続き全体最良。
- 2026-07-05 に `exp183_selector_confidence_replacement_only_on_exp148` を `exp194_exp183_selector_confidence_replacement_only_on_exp148` として Kaggle train v1 完了。exp188 add-only negative の競合切り分けとして isolated replacement-only を評価したが、`lgb_mean` 9.329893102 で exp148 8.501281182 から +0.828611921 悪化した。feature join coverage は pass だが、`learned_likelihood_confidence` block を exp183 selector confidence block へ置換する仮説は不採用。current-test feature generation / inference port / submit はしない。backlog 完了として外す。
- 2026-07-05 に `denoised_calibrated_matching_replacement_only_on_exp148` / `exp195_denoised_calibrated_matching_replacement_only_on_exp148` の Kaggle train v1 を完了した。exp190 add-only は `lgb_mean` で小幅 negative だが `lgb1` 単体は改善したため、exp148 の `learned_likelihood_confidence` (`ll_*` 54列) との競合を切り分ける replacement-only として、active variant は `projection_correction + u_disagreement + denoised_calibrated_matching` のみにした。結果は `lgb_mean` 9.409612611 で exp148 8.501281182 から +0.908331429、exp190 add-only 8.503596159 から +0.906016451 悪化し、全 single config も悪化した。DCM block は exp145 learned likelihood confidence block の代替にならないため、current-test feature generation / inference port / submit はしない。backlog は完了/不採用として外す。
- 2026-07-03 に `exp178_supervised_gr_window_matcher_from_known_tvt_prefix` の Kaggle train v1 を完了した。known `TVT_input` prefix row から 102,400 pair rows / 10,240 anchors / 160 wells の real-GR / shuffled-GR / no-GR supervised window-pair smoke を作成。real GR logistic は pair AUC 0.765413549、shuffled GR logistic 0.662345939 で +0.103067610 上回り、top1 within10 も real GR 0.355957031 / no-GR 0.252929688 で +0.103027344。real GR expected-error は AUC 0.827294、top1 within10 0.513672、top5 coverage 0.959961 とさらに強い。したがって `supervised_gr_window_matcher_from_known_tvt_prefix` は完了として backlog から外し、次は direct TVT replacement ではなく learned GR match probability / expected-error / margin / entropy を PF/Beam candidate confidence feature または exp148/exp092 add-only feature として評価する。
- 2026-07-04 に `exp180_learned_gr_window_matcher_features_on_exp148` の Kaggle feature cache と CPU split train 3本を完了した。feature cache は 3,783,989 rows / 773 wells / 61 GR matcher features、train は 355 features / 15 boosters。pooled OOF は `lgb0` 8.554800138、`lgb1` 8.581198811、`lgb2` 8.577998518、3-model `lgb_mean` 8.514526367。exp148 `lgb_mean` 8.501281182 から +0.013245185 悪化したため、`learned_gr_window_matcher_features_on_exp148` は完了/不採用として backlog から外し、inference port / submit はしない。pair smoke の GR signal はあるが、exp148 global add-only feature としてはノイズが勝つため、同じ設計の追加拡張はしない。続ける場合は PF/Beam selector 側の candidate confidence / uncertainty feature に限定する。
- 2026-07-04 に `exp185_last50_first_prefix_feature_rebuild_on_exp148` の Kaggle feature cache と CPU split train 3本を完了した。feature cache は 3,783,989 rows / 773 wells / 76 last50-first prefix rebuild features、train は 334 features / 15 boosters。pooled OOF は `lgb0` 8.636150399、`lgb1` 8.583238238、`lgb2` 8.583791509、3-model `lgb_mean_split3` 8.544817143。exp148 `lgb_mean` 8.501281182 から +0.043535961 悪化したため、`last50_first_prefix_feature_rebuild_on_exp148` は完了/不採用として backlog から外し、inference port / submit はしない。exp161/166/172/185 を通して last50 prefix crop-window 系は exp148 ML anchor を超えないため、この方向の追加拡張はしない。
- 2026-07-03 に `exp179_cnn_sdf_mtp_heatmap_probe` の Kaggle train v2 / T4 GPU を完了した。v1 は P100 割当で Kaggle PyTorch 2.10 が `sm_60` 非対応のため失敗し、v2 で `machine_shape=NvidiaTeslaT4` を明示した。discussion 699853 準拠の 5ch heatmap (`t_gr`, `h_gr`, `t_gr-h_gr`, observed `TVT_input` history SDF, mask) と K=10 path head を、target-free flat prior window center で 1 fold / small wells smoke として学習した。valid 512 samples / 32 wells、target-in-grid rate 1.0。`real_gr` は top3 within10 0.44921875、top10 0.794921875、top10 oracle RMSE 14.071006。`shuffled_gr` は top3 0.232421875、top10 0.541015625、`no_gr` は top3/top10 0.0625。real GR は shuffled-GR を top3 +0.216796875、no-GR を +0.38671875 上回り、GR signal を使えている smoke と判断する。したがって `cnn_sdf_mtp_heatmap_probe` は完了として backlog から外す。ただし 1 fold / fixed 128x64 window の診断なので、direct TVT replacement、inference port、submit はしない。次は full-fold / larger-window / geometry-channel ablation に限定する。
- 2026-07-03 に `exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe` の Kaggle train v1 / T4 GPU を完了した。24 CNN models、773 usable wells、full-fold `base_real_w128_b64_fullfold` は top3 within10 0.500000、top10 0.808908、top10 oracle RMSE 13.296284。`base_shuffled_w128_b64_fullfold` は top3 0.218536、`base_no_gr_w128_b64_fullfold` は 0.071429 で、real GR は shuffled-GR を +0.281464、no-GR を +0.428571 上回った。したがって 5ch heatmap CNN/SDF/MTP の GR signal は full-fold でも支持される。一方 `geometry_real_w128_b64_fullfold` は top3 0.487710 で base より -0.012290、`geometry_real_w256_b96_fold01` は top3 0.417512 と弱く、geometry channel / larger window は現設定では採用しない。worst-well top3 0.0 が残るため、full-length inference、direct TVT replacement、softmax average、PF weight replacement、submit はしない。`cnn_sdf_mtp_heatmap_fullfold_geometry_probe` は完了として backlog から外し、続ける場合は heatmap path 出力を selector / confidence feature に限定する。
- 2026-07-02 に `exp162_learned_likelihood_rank_slot_on_exp148` の Kaggle CPU split train、hidden-safe inference v4、code submission を完了した。exp098 の rank-slot U-shape 表現を exp145/148 learned probability / predicted-error 順位で作り直し、CPU runtime timeout を避けるため `lgb0` / `lgb1` / `lgb2` train notebooks に分割した。split pooled OOF は `lgb0` 8.488049241、`lgb1` 8.456600574、`lgb2` 8.443346041 で、single model CV では exp148 `lgb_mean` 8.501281182 を上回った。一方、hidden-safe inference v4 の code submit `ref=54247043` は Public LB 8.100 で exp148 7.960 より +0.140 悪化した。したがって `learned_likelihood_rank_slot_on_exp148` は完了/不採用として backlog から外し、exp148 を ML route submitted anchor として維持する。CV 改善が LB に転移しなかったため、rank-slot 系は単純 add-only ではなく、必要なら split OOF の by-well / bucket readout に限定して原因確認する。
- 2026-07-04 に `corr_prune_sanity_readout_on_exp148` を `studies/feature_replacement_audit/corr_prune_sanity_readout.py` として実装し、no-training readout を生成した。保存済み exp148 correlation audit、exp148 train/inference schema、feature importance、exp145 train/rawtest schema、生成元コード参照だけを読み、Kaggle GPU 学習・推論・提出は行っていない。出力は `studies/feature_replacement_audit/outputs/corr_prune_sanity_readout_on_exp148/`。exact prune 17 列は `drop_exact_replacements_17` として YAML/JSON fragment に固定し、formation last50 12 列、learned-likelihood slim review 4 列、U-projection slim review 14 列は別 bucket に分離した。exp148 train/inference schema diff と exp145 train/rawtest schema diff はどちらも non-both 0。これは OOF 改善を主張するモデル実験ではなく、後続 `exact_replacement_prune_on_exp148` の列名取り違えと parity 漏れを防ぐ安全装置として完了扱いにする。
- 2026-07-05 に `exact_replacement_prune_on_exp148` / `exp198_exact_replacement_prune_on_exp148` の Kaggle train v1 `kentookumura/exp198-exact-replacement-prune-exp148-train` を完了した。active variant は `drop_exact_replacements_17` のみ、3 LGB configs x 5 folds = 15 boosters、control / parent 再学習なし。3,783,989 rows / 773 wells / 277 features、feature join coverage pass、削除対象 17 列の schema 残存なし。pooled OOF は `lgb0` 8.525098952、`lgb1` 8.531602621、`lgb2` 8.476691203、`lgb_mean` 8.457923653 で、exp148 GPU train `lgb_mean` 8.501281182 から -0.043357529 改善した。`000_050` / `050_100` / `1000_plus` は改善、mid bucket は小幅悪化、well 単位は 423 改善 / 350 悪化、最大悪化は `b37fd114` +1.022149086 RMSE。inference v4 `kentookumura/exp198-exact-replacement-prune-exp148-inference` も完了し、14,151 rows、fallback 0、submission SHA256 `e5b71f6f576a62567adfe189c2def12a7720375e264ce8c66b31456db7848c36`、submit-check PASS。scoring ref `54354847` は Public LB 7.930。exp148 GPU inference v7 7.960 と exp193 7.946 は上回ったが、exp148 CPU runtime anchor 7.921 には届かないため未採用。元の `exact_replacement_prune_on_exp148` backlog は完了として外す。

- 2026-07-05 に `typewell_hard_window_pct40_base_surface_keep_exp145_ll_on_exp148` / `exp199_typewell_hard_window_pct40_base_surface_keep_exp145_ll_on_exp148` の Kaggle train v1 を完了した。exp148 downstream ML surface の base 196 / projection / U-disagreement を exp196 pct40 hard-window cache に差し替え、exp145 `ll_*` を残す混合 provenance 診断。3,783,989 rows / 773 wells / 294 features / 15 boosters、feature join coverage pass。pooled OOF は `lgb0` 8.551067731、`lgb1` 8.533458032、`lgb2` 8.570960612、`lgb_mean` 8.496204218 で、exp148 GPU `lgb_mean` 8.501281182 から -0.005076964 の小改善だった。`lgb2` は悪化し、改善幅も小さいため direct inference / submit はしない。backlog は完了扱いで外した。clean regeneration は実装コストに対して追加価値が薄いと判断し、`typewell_hard_window_pct40_base_surface_regen_ll_on_exp148` もユーザー指示でバックログから削除した。
- 2026-07-09 に `lgb_oof_gaussian_emission_hmm_on_exp148` / `exp221_lgb_oof_gaussian_emission_hmm_on_exp148` の Kaggle train v3 を完了した。v1 は notebook `kernelspec` 欠落で起動前 ERROR、v2 は 3 variants で 12h timeout したため、v3 は partial logs で最良だった exp148 `lgb_mean` x `sigma=20/lambda=0.50` の single variant に絞った。3,783,989 rows / 773 wells / 0 boosters、runtime 17,827.454 sec。train-side OOF RMSE は 8.327736951 で、exp148 `lgb_mean` 8.501290984 から -0.173554033、exp193 `lgb_mean` 8.456676053 から -0.128939102 改善した。全 distance bucket と exp115 hidden-like subgroup も改善し、step-delta spikes は 0。一方、exp148 比 264 wells / exp193 比 278 wells は悪化し、最大悪化は `2e63d9de`。同日 inference v1 `kentookumura/exp221-lgb-hmm-exp148-infer` も完了し、current-test exp148 `lgb_mean` を notebook 内で生成して HMM に渡す hidden-safe 経路で 14,151 rows、fallback 0、submit-check PASS、submission SHA `d90926bc87268285640863ddc3e24fbaa4d715c1b7394f7410a2d4f6d13b7cc3`。submission ref `54490473` は Public LB 7.953。exp148 GPU 7.960 は小さく上回ったが、exp193 7.946 / exp148 CPU 7.921 / exp218 7.843 には届かず採用しない。CV 改善が LB に転移しにくかったため、続ける場合は fixed sigma ではなく quantile band / uncertainty-calibrated sigma 側を検討する。
- 2026-07-09 に `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction` の Kaggle train v1 / inference v1 / code submission を完了した。Connor Tynan 公開 notebook `/tmp/kaggle-notebooks/connortynan-k16-versioned/rogii-k16-spline-kernel-knn-adaptive-kappa.py` の deterministic v6 fallback を source-port し、K=16 segment spline、raw/smoothed donor field、XY local-linear kNN、adaptive kappa、near-strike ANCC local theta、typewell GR correction、U-projection を CPU-only / 0 boosters で実行した。v7 neural committee / v8 GBM meta-layer は external weights 不在のため Stage 1 では無効化。train は target well を donor field / kappa fit / ANCC surface sample から除外する 5-fold group-safe CV で、RMSE 9.427109596、MAE 6.148527797、within10 0.807709536、OOF rows 3,783,989、OOF decompressed SHA `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`。inference v1 は 14,151 rows、submission SHA `b71e15f7dc7e66f7be70db4a81d9ec72e1001ff2ba13907c3aba24938e906047`、submit-check PASS。Code submission ref `54491603` は Public LB 9.837 で、exp206 からは大幅改善したが exp218 ML anchor 7.843 / exp148 CPU runtime 7.921 / exp082 ensemble 7.601 に届かないため不採用。

- 2026-07-13 に `two_regime_rate_noise_pf` / `exp242_two_regime_rate_noise_pf` のKaggle CPU train v2を完了した。v1は`turn_multiplier`のcall-site欠落でPF前に失敗し、科学設定を変えない最小修正でv2を実行した。fixed transition `[[0.9998,0.0002],[0.02,0.98]]`、初期495 smooth / 5 turn、turn rate noise 4倍、500 particles / 128 seedsの単一variantは3,783,989 rows / 773 wellsを23,665.002秒で完走したが、RMSEはexp072 `likpf_mean` 11.594898から13.254455へ+1.659557悪化した。全distance bucket、hidden-like spatial +0.864215、typewell-purged +0.906064も悪化し、275 wells改善 / 498悪化、最大回帰+41.956968。turn particle fraction 0.018088に対しposterior mass 0.017897で、GR likelihoodが高noise regimeを平均的に支持した証拠もない。したがって完了/不採用としてbacklogから外し、transition・初期比率・multiplierの追加grid、raw-test inference、submitは行わない。後続ではtail中のdynamic high-noise注入より、known prefixだけから作る離散的な初期rate候補をbaseと並存させる`multi_scale_initial_rate_candidates`を優先する。

- 2026-07-15 に `pf_seed_medoids` / `exp243_pf_seed_medoids` のv3 full single-notebookを完了した。旧4 shardはreplay差RMSE 0.743077とsource SHA分裂で棄却したが、PF入力float32 round-trip除去、canonical exp072/exp209 SHA分離、seed `+1`を揃えたv3は3,783,989 rows / 773 wellsでsaved `likpf_mean`との差がmean absolute / RMSE / maximum absoluteすべて0、全well `ok`、runtime 37,067.406秒。最良direct medoid `K3 m0`はRMSE 12.296667でbase 11.594898から+0.701770、344 wells改善 / 429悪化、worst +20.953998、全distance/hidden-likeも悪化したためdirect replacementは不採用。一方、exp237 base8 + K8 oracleはrow -1.348387、block128 -1.405104、block256 -1.372076、block512 -1.316683、whole-well -1.092839で、374 wellsを改善し、K8 medoid unique-bestは43.8800%。all-KはK8単独からwhole-well -0.006406だけなので、candidate generation仮説は支持してK8に縮約する。target-free選択規則未確立のためraw-test inference、selector、submitは行わず、保存済み候補のselectability auditを次候補とする。

- 2026-07-15 に `pf_seed_medoid_selectability_audit` / `exp252_pf_seed_medoid_selectability_audit` のKaggle CPU train v1を完了した。exp243 v3保存済みbase8 + K8、cluster manifest / summary、PF diagnosticsをSHA固定入力とし、PF再実行・学習なしで3,783,989 rows / 773 wells、5 scope、bank 10 score / candidate 7 scoreを86.053秒で監査した。K8内ではlikelihood mass / rank / gapが5/5 scopeでshuffled AUCを上回り、whole-well AUCは0.675214 / 0.655102 / 0.654235でcandidate順位付け信号を部分支持した。一方、bank最良`resampling_rate`はwhole-well AUC 0.560593に留まり、likelihood-mass top1もuseful 374 wellsのcoverage 0.516043、union-best match 0.280749、best base8比loss平均+3.194947 ftだった。したがってbase8を捨てるbank gate、fixed top1、3 score単独selector、現時点でのraw-test PF再生成、inference、submitは進めない。likelihood mass / rank / gapは、base8 fallbackと別bank gateを持つfold-safe二段selector、または既存`topk_path_confidence_features`へのadd-only candidate-ranking特徴量候補として残す。生成コストは保存済み候補のreadoutが86.053秒 / 773 wells、raw PF seed bank + medoid生成がexp243 v3実測37,067.406秒 / 773 wells（約10時間18分）。hidden test約200 wellsの単純比例は約2時間40分だが未実測の参考値とする。

- 2026-07-17 に `pf_ancc_pf_z_multiseed_stability_audit` / `exp266_pf_ancc_pf_z_multiseed_stability_audit` のKaggle CPU train v3を完了した。exp072 exact PF ANCC / PF-Zを各600 particles × 64 seedsで3,783,989 rows / 773 wellsへ再生成し、seed 0は両手法とも全行差0、runtime 12,482.144秒。`11d0f5ac`は新規63 seedでもHMM / likelihood-PFへのstrong marginを両手法100%再現し、RMSE 5 ft以下はPF ANCC 98.4%、PF-Z 100%。PF ANCC元seedはpercentile 0.508で典型、PF-Z元seedは0.952でむしろ悪い側だったため、単一seed偶然仮説を棄却する。一方、元seedstrong 53 wellsで両手法の過半数seedがstrongなのは21、80%以上再現11、全seed再現4、両手法で80%以上のseedがRMSE 5 ft以下なのは`11d0f5ac` / `fb0904bd`の2 wellsだけ。PF ANCC strong groupは元seedが良い側へ偏り、元seedRMSE平均9.135に対し新規seed中央値平均12.075でselection-on-seedを確認した。raw特徴に明瞭な単一triggerはなく、tail長はseed誤差/分散を弱く増やす一方、strong再現率はHMM / likelihood-PFの失敗度とより強く連動する。64 seed meanはPF ANCC 14.493051→12.830319、PF-Z 17.788171→17.074522だが、直接inference/submitはしない。実装済みbacklogは完了として外し、PF ANCC固定4/8 seed meanの低コストcandidate監査だけを次候補へ残す。

- 2026-07-14 に `shrinkage_residual_scale_emission_hmm_on_exp218` / `exp240_shrinkage_residual_scale_emission_hmm_on_exp218` を完了し、ユーザー判断で方向性をclosedとした。exp218-center exact HMMのsigmaだけを変え、scalar `sigma=20`、variance shrinkage alpha `0.25 / 0.50`を1 version 1候補で比較した。RMSEは`8.361307776 / 8.351122273 / 8.336863897`でalpha 0.50が有限grid最良だったが、alpha 0.25比でMAE`+0.041424`、within10`-0.001094`、hidden-like 2群が悪化し、352 wells改善 / 421悪化だった。行別sigmaの部分反映には小さいoverall効果がある一方、well一般化の安定性を支持しないため、追加alpha grid、同一sigma推定の再調整、raw-test inference、submissionは行わない。実装待ちbacklogから削除し、結果はexp240の記録に保持する。

- 2026-07-14 に `negative_space_gr_barrier_audit` / `exp246_negative_space_gr_barrier_audit` のKaggle CPU train v2を完了した。v1は773-well本体後のgzip下位buffer close漏れでSHA段階に失敗し、科学設定を変えないflush/close修正だけでv2を再実行。773 wells / 3,783,989 rows、runtime 733.672秒、1 diagnostic variant / LightGBM 0 config / fold 0 / booster 0。5 safety guardsはすべてfailし、true-path瞬時違反率0.006422、anchor-component survival 0.991760、good-candidate false-prune 0.361574、union oracle 7.434021 -> 9.090919（+1.656898）、worst-well `d07aed8f` +77.747616。hidden-like 2面も+2.189781 / +2.204004悪化し、well単位は改善0 / 同値138 / 悪化635だった。`likpf_mean` / `pf_ancc`のbad-candidate precision liftは1.394x / 1.301xだが、hard-history pruneではgood candidateも31.2% / 28.4%切る。したがってexp246で実装した「official evaluation tail全体を1枚のglobal surfaceにし、最初の違反を`valid_after_history`で末尾まで累積するhard barrier」、そのthreshold grid、HMM/PF/Beam edge cut、raw-test inference、submitはclosedとする。後続レビューで、ユーザーが着想時に見ていた局所segment heatmapとはwindow、正規化、corridor resetの契約が異なると判明したため、segment-local版は当時未検証の別仮説としてbacklogへ戻し、後続exp250で別途検証した。exp246のglobal瞬時endpoint/crossing/barrier fractionは既存`topk_path_confidence_features`の材料に限定する。

- 2026-07-15 に `segment_local_negative_space_gr_corridor_audit` / `exp250_segment_local_negative_space_gr_corridor_audit` のKaggle CPU Stage 0 / Stage 1を完了した。既存exp249とは分離し、MD 256 ft / stride 128 ft / 4 ft grid / flat-Z ±256 ft / minimum-bottleneck `tau_star+0.25`を固定、real/shuffled 2 surfaces、model config / fold / booster / PF再生成は0 / 0 / 0 / 0で773 wells / 3,783,989 candidate rowsを7,633.823秒で監査した。Stage 0 manual parityはPASSしたが、Stage 1 guardは2/8 PASSに留まった。pooled real AUC 0.530134、shuffled差+0.035934、q90 lift 0.776971、good false-alert 0.232020、overlap path差median 57.61 ft / p90 258.684 ft、risk Spearman 0.448723、hidden-like AUC 0.531044 / 0.532323、by-well false-alert p95 0.757381 / max 0.984733。truth coverage差+0.059580とreal-shuffled AUC差だけはPASSしたが、candidate誤りの識別・誤警報・segment overlap・well一般化は不十分だった。0–100 ftのreal AUCは約0.82だがshuffledも約0.77で、支配的な1000+はreal AUC 0.515575だったためdistance/base-error交絡を含む。契約どおりhard利用、threshold/slack/segment grid、direct path、`topk_path_confidence_features`への追加、raw-test inference、submissionをclosedとする。再訪は保存済みcandidate-segment artifactだけでnear signalを距離条件付きに帰属する低優先readoutに限定する。

- 2026-07-15 に `segment_local_corridor_near_bucket_signal_attribution_readout` / `exp256_segment_local_corridor_near_bucket_signal_attribution_readout` のKaggle CPU train v1を完了した。exp250 Stage 1のcandidate-segment / group / by-well / summaryをSHA固定入力とし、Stage 1・corridor・candidate・modelを再実行せず、291,710 rows / 145,855 paired keys / 773 wellsを6.486863秒で集計した。0–100 ft pooled AUCの2 bucket weight平均はreal 0.819846 / shuffled 0.773559だったが、distance x candidate-family条件付け後は0.598678 / 0.574742へ下がり、real AUCのcross-family寄与は0.221168だった。near weightは全体の1.048546%、10 family-bucket strata中AUC算出可能は6 strata / 4 familiesに限られた。family x well条件付きAUCは0.522220 / 0.511096（差+0.011124）、positiveは1,138 / 2,330 strata、pair-mass share 0.522241で正負ほぼ半々。pooled q90は両controlとも1.0、risk=1 weight比はreal 0.188251 / shuffled 0.270472だった。nearの弱いGR固有差は残るが、pooled約0.82の大部分はcandidate-family base rateへ帰属し、broadなwell一般化signalではない。exp250 hard use / feature化 / near rule / candidate変更 / parameter grid / raw-test inference / submitはclosedのままとし、`topk_path_confidence_features`にも混ぜない。原因切り分けは完了したため新規backlogは追加しない。

- 2026-07-15 に `exp239_distribution_matched_multicut_pseudotail` のfull exp218 augmentationを完了した。distribution-matched early pseudo-tail 799,961 rowsをweight 0.5でofficial 3,783,989 rowsへ追加し、official-only GroupKFoldで3 configs x 5 folds = 15 boostersを学習したが、OOFは8.697380066で保存済みexp218 8.475793752から+0.221586314悪化した。48 cache shards、380-feature schema、valid-well pseudo除外、SHAは全て契約どおりで、OOMもmemmap streamingにより解消したため、実装失敗ではなく直接データ拡張仮説を不採用とする。ユーザー指定trial submission ref `54720769`もPublic LB 7.944でexp218 7.843より+0.101、exp238 ML anchor 7.775より+0.169悪化し、CVとLBの方向が一致した。weight微調整や同方式の追加提出は行わない。派生`exp244`はearly-only悪化を前提に、late viewが独立に補償する証拠なしで統合効果を主張しない。
- 2026-07-16 に `bidirectional_prediction_start_pseudotail_augmentation` / `exp244_bidirectional_prediction_start_pseudotail_augmentation` の本来のearly / official / train-only late統合学習を完了した。official 3,783,989 rows weight 1.0へ`-1000/-250/+250/+1000`の3,081 views / 770,157 rowsをweight 0.5で追加し、exp218-family 3 configs x 5 folds = 15 boostersを学習した。OOFは8.475793752 -> 8.472379731（-0.003414021）、1000+ -0.009135、hidden-like spatial -0.415836、typewell-purged -0.405110、改善3/5 foldsだった。一方、387 wells改善 / 386悪化、14 wellsが+2 ft超悪化し、worst `059c8f24`は+16.650567、fold 1は+0.909638だった。380-feature schema、124 pseudo shards、outer-valid source-well除外、15 model、artifact SHAは全contractをpassしており実装失敗ではないが、worst-well guard失敗により`adoption_supported=false`とする。mixed weight grid、inference、submissionは行わない。early/late同時投入なので方向別寄与は未識別であり、再開は同じsampling/weightのmatched early-only / late-only attributionを別途承認して行う場合に限る。
- 2026-07-16 に `matched_early_late_attribution_on_exp244` / `exp260_matched_early_late_attribution_on_exp244` のKaggle GPU train v1を完了した。exp244のcache / sampling / pseudo weight 0.5 / 380 features / fold / 3 configsを固定し、early-only `-1000/-250`とlate-only `+250/+1000`を各15 boosters、合計30 boostersで分離した。early-onlyは8.513933814（exp218比+0.038140063）、1000+ +0.036480、hidden-likeは-0.332981 / -0.325111、改善2/5 folds、worst `059c8f24` +18.623158。late-onlyは8.489116155（+0.013322404）、1000+ +0.013735、hidden-like +0.052784 / +0.058461、改善2/5 folds、worst `7850c72e` +3.408451だった。hidden-like改善と`059c8f24`崩壊はearly側へ帰属し、同wellのlate-onlyは-0.434122なのでlateは崩壊源ではない。late-onlyはearly-onlyよりoverallで0.024818良く安全だが、raw exp218にも届かず全必須guardを失敗したため独立補償を否定する。mixed exp244の小gainは方向単独で再現せず非加法的相互作用を示唆するが、mixed自身もworst guardを失敗している。両方向を不採用、weight / offset grid、risk gate、inference、submissionをclosedとし、prediction-start augmentation branchを終了する。原因分離は完了したため救済backlogは追加しない。
- 2026-07-15 に `missing_gr_masking` / `exp247_missing_gr_masking` のKaggle CPU train v1を完了した。exp221 train v3の保存済み補間controlとexp148 OOF LGB unary `sigma=20/lambda=0.50`を固定し、raw evaluation GR欠損rowのGR emissionだけを0にする1変更ablationを773 wells / 3,783,989 rows、11,409.172秒で監査した。overall RMSEは8.327728213 -> 8.322894658（-0.004833555）だがMAEは+0.042731938、short missing 1-31もRMSE -0.003848864に対しMAE +0.067176896。hidden-like spatialはRMSE +0.005961600、typewell-purgedは-0.004869435、well別は改善386 / 悪化387、worst `c66be2b8`は+2.576980644悪化した。finite coverageは両候補100%。tinyで非一貫なRMSE gainとtail regressionのため一律maskは不採用とし、run-length gate/grid、raw-test inference、selector、submitへ進まずbranchをclosedとする。

- 2026-07-15 に `numba_allseed_pf_speed_reproduction` / `exp254_numba_allseed_pf_speed_reproduction` のKaggle CPU probe v1を完了した。target-free固定3 wells / 14,450 eval rows、particles 500、seed数`1/4/16/32/64/128`、candidate spec数`1/10/100/300`、1 diagnostic variant / LightGBM 0 config / fold 0 / booster 0で、wall runtime 436.888720秒、peak RSS 683.09375 MiB。trajectory / log-likelihood / mean / ESS / resampling、exp243保存mean、repeat / cache / warm SHAの全guardがexactだった。128-seed PF core合計はlegacy 80.897349秒、all-seed 81.754755秒で`legacy/all-seed=0.98951x`となり、seed loop全体のJIT化だけには速度上の採用根拠がない。single-seed本体は既にNumba compiledで、Python call overheadがPF本体に対して小さく、外側loopのJIT化だけでは計算量が減らないことが原因と考える。cached seed bankから300 candidateを再集約するwarm generationは3 wells合計0.104562秒だったが、固定集約1本を使う通常推論には不要で、exp252のseed-medoid selectability gateも弱いため用途がない。773-well all-seed + warmは21,436.315秒（約5時間57分）の行数比例projectionであり、追加実測の価値もない。all-seed高速化、cached 300-candidate基盤、full workload、追加candidate探索、raw-test inference、submissionをすべて不採用とし、ユーザー指示によりbranchをclosedとした。新規backlogは追加せず、元backlogも完了・不採用として外す。

- 2026-07-16 に `prefix_verified_bounded_candidate_controller` / `exp253_prefix_verified_bounded_candidate_controller` のKaggle CPU Stage 1を完了した。Stage 0のsorted 32 wellsではoverall -0.198601、1000+ -0.226789、4/5 folds改善だったが、固定設定をstable SHA256 modulo 4の4 shardで全773 wells / 2,319 requestsへ広げたrow-level aggregateでは、exp238 base 7.936701 -> controller 8.205455（+0.268755）へ悪化した。000-250 ftは-0.002746〜-0.012375と小改善した一方、1000+は+0.307983、hidden-like 2面は+0.282873 / +0.267543、foldは0/5改善。361補正wells中150改善 / 211悪化、worst `fcfcc902`は+10.310641だった。technical checksは9/9通過しており実装失敗ではない。ユーザー指定どおりworst-wellと補正well勝率をmonitor-onlyとしても、他の必須guardが不通過のためこのprefix評価 + bounded correctionを不採用とし、parameter grid、inference、submissionへ進めずbranchをclosedとする。新規backlogは追加しない。

- 2026-07-15 に `nested_selector_gated_bounded_direct_readout_on_exp238` / `exp255_nested_selector_gated_bounded_direct_readout_on_exp238` のKaggle CPU train v1を完了した。exp238 add-only OOF 7.936690をfallbackに、outer-valid selector scoreのgain/marginとtarget-free well consistencyでgateし、top1方向へ最大4/7.5/12 ftだけ動かす固定3 profileを3,783,989 rows / 773 wellsで監査した。model config / fold / boosterは0 / 0 / 0、selector score 5 foldのSHAと全行coverageはPASS、truthはgateに不使用。hard top1は8.512262（+0.575572）、conservativeは7.938384、balancedは7.929965、assertiveは7.877990（-0.058700）。assertiveはnear、1000+、hidden-like 2面、3/5 foldsも改善したが、604補正wellsの324改善 / 280悪化、106 wellsが+0.25 ft超悪化、worst `d7ba4f9d` +3.151245でguard fail。fixed consistency gateではwell-tail riskを抑えられないためinference / submitは行わず、元backlogは完了として外す。clip緩和・alpha増加・hard top1はclosedとし、再訪はouter-train-only well-risk discriminatorに限定する。後続のユーザー訂正により、exp255は本来依頼された「LightGBM既存selector出力slotの置換」ではなく別仮説だったと確定した。履歴は保持し、正しいreplacement-only実装はexp257へ分離する。
- 2026-07-16 に `nested_selector_output_replacement_only_on_exp218` / `exp257_nested_selector_output_replacement_only_on_exp218` のKaggle GPU train v1を完了した。exp218の380列schemaを固定し、既存`ll_*` 54列のうちselector出力29列だけをexp238 nested selectorの11候補scoreから上書き、入力診断25列を維持、`nsel_*`追加0とした。保存済みselector scoreを使いselector再学習0、finalは3 configs × 5 folds = 15 boosters、control再学習0。`lgb_mean`は8.101331で同一fold exp238 add-only 7.936690から+0.164641悪化し、near +0.068730、1000+ +0.184078、改善fold 1/5、worst-well +13.291303でguard fail。380列schema・29/25 slot契約・15 model・artifact SHAはPASSしているため実装失敗ではなく、11候補情報を既存29 slotへ圧縮する今回のreplacement-only仮説をtrain-sideでは不採用とする。元backlogは完了として外す。その後のユーザー明示指示によりhidden-safe inference v2とcode submissionを完了。14,151行、context 184列、COPCF 41/41、欠損context 0、test-test neighbor 0、29列上書き / 25列維持 / `nsel_*` 0 / 最終380列、保存済み20 selector / 15 final models、再学習0、submit-check PASS。ref `54753824`はPublic LB 7.718でexp238 hidden-safe 7.775を-0.057改善し、Public-LB上のML submitted anchorを更新した。ただしCV/LBの方向が反転しているためguard falseを撤回せず、train-side採用とは分離する。
- 2026-07-16 に `gr_residual_noise_transplant_augmentation` / `exp258_gr_residual_noise_transplant_augmentation` のKaggle CPU selector train v1を完了した。typewell affine再構成で説明できない実測horizontal GR residual block / missing maskをinner-train donorだけから移植し、duplicated inner-train行のbase5 `multiobs_*`を再計算した。11 candidates / 184 context、outer5-inner4の20 selector boosters、3,783,989 rows / 773 wells、runtime 16,612.296秒。residual auditは5,092,255 raw rows、20 fold contractでdonor/validation overlap 0、validationはclean、20 model完全被覆。historical exp238比でexpected-error MAEだけ4.532978 -> 4.523354（-0.009625）へ改善したが、global +0.004554、near +0.001652、1000+ +0.006871、candidate AUC -0.000175、worst-well +0.322063と5/6 guardがfailした。実装失敗やfold leakageではなくaugmentation仮説を不採用とし、契約どおりfinal TVT LightGBM 15 GPU boosters、controls、inference、submissionは実行しない。augmentation比率やblock長gridで救済せず、元backlogは完了として外す。

- 2026-07-15〜16 に `raw_test_safe_dual_objective_candidate_ranker` / `exp251_raw_test_safe_dual_objective_candidate_ranker` を完了した。旧130列版v2はfixed Viterbi 8.402085596でoverall/1000+/hidden-likeを通したがworst `fb03ae90`だけFAIL。その後、除外167列中165列の`copcf_*`はtest生成不能ではなく未実装だったと訂正し、trainはcross-fit/OOF、raw testはtest wellをsource poolから全除外したfull-train typewell/spatial referenceとして再生成した。v3/v4 auditは297 parent / 295 selected / 165 regenerated、41 raw-test base `copcf_*`、test-well source overlap 0、hard check全PASS。v4は1 variant / 2 config / 5 folds / 10 CPU boosters、parent/control再学習なしで完走したが、fixed Viterbi 8.502212005でexp218比+0.026418253、exp248 original-only比+0.080796908、1000+ 9.326545505、worst `fb03ae90` 58.004236030となりoverall/1000+/worstの3 guardがFAILした。raw-test `copcf_*`再生成契約の技術的成立だけを採用し、295列rankerは不採用、inference / submissionは行わない。exp259のfixed clean controlとして保存metricsを使う。
- exp238 raw-test copcf parity v1は41列の生成可能性を示した一方、保存summaryでは各visible test well ID自身がfull-train spatial KNNに含まれ、typewell neighbor数も自己ID込みの14/41/14だった。visible test IDsはtrainにも存在するため、同runをleakage-safe parityの証拠として扱わない。exp251は3 test IDsをsource curve・typewell matching・cluster geometry・spatial KNNから全除外し、typewell source 12/40/12と最終source overlap 0をhard guardにする。
- 2026-07-16 に `coordinate_equivariance_path_warp_augmentation` / `exp259_coordinate_equivariance_path_warp_augmentation` のKaggle CPU exact datum train v1を完了した。事前auditで773/773 rejectされた`md_stretch`を除外し、exp251の295列schemaを固定、clean rowsを保持したままstable 25% wellsへexact TVT datum viewだけをouter-trainに追加した。1 variant × 2 objectives × 5 folds = 10 boosters、control再学習0、5/5 equivariance guard PASS。saved exp251 clean control比でfixed-Viterbiは8.502212005 -> 8.427125551（-0.075086454）、candidate logloss -0.000026563、1000+ -0.082177424と改善したが、exp115 spatial +0.067319432、typewell-purged +0.045377256、最大well回帰`aed44918` +6.370552990で3/6 guardがFAILした。overall gainは認めるがhidden-like / well stabilityを優先して不採用、inference / submitは行わない。元backlogは完了として外し、比率・shift幅の事後gridは閉じる。再訪は1000+限定の低比率variantに限る。

- 2026-07-17 に `fixed_exp226_w500_equal_blend` をexp263 Stage 1で完了した。固定式は
  `0.50*exp226_k16 + 0.25*likpf_mean + 0.25*exact_hmm`、OOF RMSE 8.238331、5/5 folds改善。
  hidden-safe CPU inference v2は14,151 rows / 3 wellsを225.459秒で完走し、formula parity最大0、
  6 primitiveのreference差最大0.000484375、submission SHA
  `6316695197ee67c9a2aaa23754e6f2a5cf30dd0ec4ef1a018921f9ea640a1dbc`、submit-check PASS。
  ref `54761954`はPublic LB 7.800でexp226単体9.837を-2.037、exp218 7.843を-0.043改善したが、
  exp257 7.718より+0.082、exp082 ensemble 7.601より+0.199悪いため全体anchorは更新しない。
  固定blendの補完性と推論経路は採用し、LBを見たweight grid・係数再fit・追加submitは行わない。

- 2026-07-17 に `lightgbm_extra_trees_ablation_on_exp218` / `exp261_lightgbm_extra_trees_ablation_on_exp218` のKaggle GPU train v1を完了した。exp218の380-feature surface、well GroupKFold、3 config、seed、GPU modeを固定し、変更は`extra_trees=True`だけ。親/control再学習0、新規15 boostersを単一notebookで実行した。親3-config平均8.475793752に対し8.755217124（+0.279423372）、全3 config悪化、改善1/5 folds、1000+ +0.315774718、hidden-like 2面+0.243319/+0.250390、worst-well +11.324423。親予測parity誤差0、parameter差分、380列schema、15-model manifest、生成物SHAはPASSしたが全guard false。fixed 0.25 blendもoverall +0.031917897のため、回帰variant、inference、submit、parameter rescueを不採用とする。selector LightGBMは別surface/objectiveのexp262で独立判定する。小さなnear/mid改善の再訪は保存OOFだけの低優先0-booster安定性readoutに限定する。

- 2026-07-17 の `exp265_target_free_pairwise_candidate_divergence_soft_experts_on_exp264` は、regime raw context自体にtraining-onlyの`ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA`を含み、score readoutも無効なexp264 Stage Bに依存するため全面無効化した。block occupancy、stability、separability、calibrationをhidden-safeな診断・negative resultに使わず、「保存済みexp264 global selectorを維持する」という結論も撤回する。

- exp265完了後のtarget-free追加readoutでは、512-row full blockに限定しても候補bank range平均がwell内の序盤/中盤/終盤で`11.3233 -> 19.7139 -> 24.4863`、15 pairのabsolute gap平均が`4.8993 -> 8.7066 -> 10.8876`へ増え、全5 foldsで同方向だった。正解TVT/誤差を使わず、bank range mean/p90、effective rank、rank switch、pair gap mean/p90の6指標×3区間=18特徴をwell単位に集約した固定K=3予備診断はOOF occupancy `255/423/95 wells`、別seed assignment一致率mean `0.996` / min `0.987`。low/mid/high divergence clusterのrange中央値はそれぞれ`5.99->9.98->13.11`、`11.16->19.98->24.21`、`21.83->39.55->51.87`だった。これはtarget-freeなwell構造の証拠だがcandidate誤差差の証拠ではないため、旧block-level `block_length_invariant_pairwise_regime_reaudit`を削除し、well区間署名をouter-foldで監査する高優先backlogへ置き換える。

- 2026-07-17 に `well_segment_candidate_divergence_signature_cluster_on_exp265` / `exp267_well_segment_candidate_divergence_signature_cluster_on_exp265` のKaggle Stage A version 2を0 boosterで完了した。3,783,989 rows / 773 wells / 18 target-free fixed-thirds features、fallback 0、別seed一致率min 0.954839はPASS。一方、pooled K=3 occupancyはlow/middle/high=`538/41/194` wellsでmiddleが基準75未満、fold 1/2/3も`7/3/9` wellsで基準10未満だった。区間別bank range low<middle<highは1/5 foldsだけ、candidate winner patternもmodal count 1でFAIL。calibration方向5/5とworst-cluster leave-one-well-outはPASSしたが、structureが再現しないため総合guardはFAIL。conditional 10 CPU boosters、downstream GPU、inference、submitは未実行でK=3 branchを閉じた。version 1のmutable exp264 input欠落はhistorical Stage B v2をprivate immutable datasetへSHA固定して修復し、version 2の全14生成物とmanifest SHAを保存した。

- 2026-07-17 の `exp272_continuous_well_divergence_risk_readout_on_exp267` は、exp263候補のactual MAEとtarget-free divergenceの関係（Spearman pooled 0.785818）だけを候補bank readoutとして保持する。exp264 predicted error由来のcalibration biasと、それを含む総合guard/negative判断はfeature availability leakageにより無効化した。

- 2026-07-18 に `raw_hmm_likpf_missing_gr_observation_neutrality_ablation` / `exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation` のKaggle CPU Stage 1 version 1を0 boosterで完了した。exp209 raw typewell-GR exact HMMの保存cacheをdecompressed SHA固定controlにし、raw GR missing evaluation rowだけGR emissionを全state 0へ変更した。3,783,989 rows / 773 wells、runtime 19,573.731秒。overall RMSEは11.938287 -> 13.348499（`+1.410212 ft`）、missing rows `+2.548257`、observed rows `+0.846115`、1000+ `+1.583805`、hidden-like 2面`+3.462999 / +3.556545`、worst well `e03b45fd` `+51.167455 ft`で全性能guardを失敗した。prediction/std finite coverage 100%、ID mismatch 0、control SHAはPASSしたため実装不良ではなく、補間GR emissionを外すblanket neutrality仮説を棄却する。likelihood-PF Stage 2、run-length gate、sigma/temperature救済、mask grid、raw-test inference、submitをclosedとする。当時はmissing-GR処理を固定した別仮説`exp270_exact_hmm_posterior_mode_candidate_audit`を継続対象としたが、後続aggregateでdirect mode候補もnegativeとなり2026-07-20に閉鎖した。

- 2026-07-18 に backlog `two_dimensional_formation_gradient_transition` / `exp273_two_dimensional_formation_gradient_transition` の2 CPU shardsとaggregate version 1（id_no `127705719` / `127705716` / `127731254`）を0 boosterで完了した。各well known prefixの`S=TVT_input+Z`へdeterministic Huber planeをfitし、gradient centerとcovariance 2軸の`+-1 sigma`から5 fixed prototypesを作り、exp209 exact HMMのposition transitionだけを`gx*dX+gy*dY+residual_rate*dMD-dZ`へ変えた。773 wells / 3,783,989 rows、saved-control parity、shard SHA、aggregate 10 CSV SHAはPASS。一方、scalar RMSE 11.938287に対しgradient 5候補は全て悪化し、best axis1-minusも12.169871（`+0.231584 ft`）。geometry-valid / turning / 1000+ / hidden-like / worst-wellもFAILした。whole-well oracleは`-0.178637 ft`残るがtarget-free選択根拠はなく、direct候補を棄却してcandidate平均、selector、raw-test inference、submissionへ進めない。唯一残したprefix-stability risk readoutもexp278でfold正方向3/5のguard FAILとなったため、formation-gradient branch全体を救済gridなしで閉じる。

- 2026-07-18 に backlog `catboost_final_regressor_swap_on_exp238` / `exp274_catboost_final_regressor_swap_on_exp238` のKaggle T4 train version 1を完了し、train-side negative resultとして閉じた。exp238の保存済み outer folds、380 base + 35 nested rank-slot features、selector score、residual target、parent `lgb_mean` OOFを固定し、Pixiux公開 `cb0` 1 configを5 folds / 5 modelsだけ学習した。raw CatBoost RMSEは8.183504でparent 7.936690より+0.246814、固定0.25 blendも7.950394で+0.013704悪化。改善はfold 2の1/5だけで、fold 4は+0.702350、1000+は+0.271067、hidden-like spatial / typewell-purgedは+0.274255 / +0.274986、worst wellは+12.293692となり全raw guardがFAILした。入力selector summary自体も`selector_guard_failed_final_train_forbidden`なので、同じ保存済みsurface上のmodel-family比較に限定して解釈する。その後ユーザー明示承認でreference-only inference v1を実行し、14,151行 / fallback 0、raw / parent / fixed0.25 blendの3出力をsubmit-check PASS。rawとparentのtest予測差はRMSE 1.270216、固定blendとparentは0.317559だった。raw CatBoost `ref=54793316`はKaggle API `COMPLETE` / Public LB 7.715で、exp257 7.718を-0.003更新するML submitted anchorとする。ただしCV/LBの方向は反転しているためtrain rejectionと不採用判断は変更せず、公開`cb1`、parameter / blend-weight grid、新しいCatBoost救済backlogは追加しない。次model-family候補だったexp275の結果は次項で確定した。

- 2026-07-18 に backlog `xgboost_final_regressor_swap_on_exp238` / `exp275_xgboost_final_regressor_swap_on_exp238` のKaggle T4 train version 2を完了し、negative resultとして閉じた。version 1（id_no `127706029`）はapproval status文字列不一致でデータ読込前に停止しbooster 0本、contractだけを修正したversion 2は1 variant / 1公開config / 5 folds / 5 models / 合計2,250 treesを2,984.807秒で完走した。exp238の保存済み415列surfaceとouter rolesを固定したraw XGBoost RMSEは8.302528でparent 7.936690より`+0.365838`、全5 foldsで悪化しfold 4は`+1.072875`。1000+は`+0.400383`、hidden-like spatial / typewell-purgedは`+0.668466 / +0.661976`、worst wellは`+13.880009`で全raw guardがFAILした。parentとの予測相関は0.999996、固定0.25 blendも7.990747で`+0.054057`悪化し、多様性も支持されない。OOF 3,783,989行、5 model SHA、fold matrix SHA、主要artifact SHAは一致確認済み。その後ユーザー明示承認でreference-only inference version 2を実行し、415.815秒、14,151行、fallback 0、5 XGBoost / 15 parent / 20 selector model、推論時学習0で完了した。rawとparentのtest予測差RMSEは0.917322、raw submissionはsubmit-check FAIL/WARN 0。正規ref `54798185`は`COMPLETE` / Public LB `7.760`で、exp238 hidden-safe 7.775を`-0.015`改善したが、現ML submitted anchor exp274 7.715より`+0.045`、ensemble anchor exp082 7.601より`+0.159`悪い。追加ref `54798337`も7.760だった。CV/LBの序列は逆転したが、train rejectionとanchor不更新は維持し、parameter grid、early stopping追加、selector XGBoost化、blend weight探索、新しいXGBoost救済backlogは追加しない。CatBoost / XGBoostのfinal-estimator family差し替え枝を閉じ、既存の高優先0-booster監査を優先する。

- 2026-07-19 に backlog `exp226_geop_centered_exact_hmm_redecode` / `exp279_exp226_geop_centered_exact_hmm_redecode` のKaggle private CPU version 1（id_no `127766774`）を完了し、negative resultとして閉じた。exp209 exact HMMを固定し、group-safe exp226 `tvt_geop`中心のGaussian unary `sigma=20 ft / lambda=0.50`を毎行1点だけ追加した。3,783,989 rows / 773 wellsを18,663.389秒で完走し、入力SHA、exp263 parity、grid / finite coverage、11 artifact SHAは全PASS。`geop_hmm`はexact HMM 11.938287を10.035987へ改善したが、promotion baseline exp263 fixed 8.238332より`+1.797655 ft`悪く、全5 foldsが`+1.248048～+2.848074 ft`回帰した。near / 1000+ / hidden-like 2面も`+0.198846 / +1.983953 / +1.381884 / +1.451614 ft`、worst wellは`+27.158481 ft`で全性能guardをFAIL。persistent-offsetの512行以内復帰率は9.0744%から11.8454%へ上がったがepisode数も551から802へ増え、固定geometry anchorは安全な復帰力にならなかった。1 HMM variant / 773 well-runs / 0 LightGBM config / 0 trained fold / 0 booster、inference/submission disabledを維持し、同じabsolute unaryのsigma/lambda/grid/process-noise救済、PF併用、直接blend/置換、raw-test inference、submitへ進めない。2026-07-19のユーザー指示により、同一unaryの救済ではない別仮説として、shift likelihoodの識別力監査、exp226座標系でoffsetだけを状態にするHMM、known-prefix masked backtestの3件を未着手backlogへ追加した。さらに別のユーザー指示により、exp279を直接救済する仮説とは分離し、修正版exp264の既存12候補を維持したまま`geop_hmm`を疎な13番目候補として追加する`geop_hmm_sparse_addonly_candidate_on_exp264`を未着手backlogへ追加した。

- 2026-07-19 に backlog `exp226_shift_likelihood_separability_readout` / `exp280_exp226_shift_likelihood_separability_readout` のKaggle private CPU version 1（id_no `127828902`）を0 boosterで完了し、positive diagnosticとして閉じた。group-safe exp226 `tvt_geop`の局所形状へ固定13 shiftを加え、exp209 raw-GR Gaussian likelihoodを非重複512行の7,787 blocksで評価した。3,783,989 rows / 773 wellsを456.972秒で完走し、row identity / finite score / bank range / quantization coverageは全て1.0。top1は0.189547 vs shuffled 0.075767、top3は0.452421 vs 0.234493、MRRは0.389626 vs 0.245536、signは0.498523 vs 0.418518で、4指標すべてが5/5 foldsでshuffledを上回り固定guardをPASSした。1000+、hidden-like 2面、persistent-offsetでも4 liftは全て正。score contractはtruth未付与、全score freeze後のcontent SHA `4a546cfe...aa46`とtruth attachment側SHAが一致し、input / gzip / 7 manifest-metric SHAも取得outputで照合した。一方top1は18.95%、persistent-offsetでも15.05%、sign絶対精度49.85%なのでdirect shift correctionは支持しない。結果はexp226座標系のslow offsetを時系列統合する別実験の検討だけを許可し、`exp226_residual_offset_exact_hmm_transition_probe`を高優先へ上げる。inference / submissionは実行していない。

- 2026-07-19 に backlog `exp226_residual_offset_exact_hmm_transition_probe` / `exp281_exp226_residual_offset_exact_hmm_transition_probe` のKaggle private CPU version 1（id_no `127831519`）を完了し、negative resultとして閉じた。`TVT_t = exp226_tvt_geop_t + delta_t`としてoffset `[-80,80] ft / step 0.35`と41 offset-rate statesだけを1 fixed exact HMMで復号し、1 variant / 773 well-runs / LightGBM config 0 / trained fold 0 / booster 0を15,042.787秒で完走した。residual-offset HMM RMSEは9.827420でexp263 fixed 8.238332より`+1.589088 ft`悪く、全5 foldsが`+0.754790～+3.041602 ft`回帰した。near / 1000+ / hidden-like spatial / typewell-purgedも`+0.280916 / +1.792419 / +1.808499 / +1.610008 ft`、worst wellは`+30.961675 ft`でoverall / fold / scope / worst-well guardをFAIL。一方、773 wells中408は改善、MAEは5.398485から5.290694、within5は0.634146から0.689860、persistent episodesは551から530へ減り、256 / 512行復帰率も`+0.000863 / +0.031897`改善した。局所GR evidenceは中心誤差と復帰を改善してもrare large offset tailを抑えられず、always-on global decoderとして安全ではない。3,783,989 rows / 773 status ok、exp263 parity、grid / finite coverage、入力SHAはPASSし、取得OOFのraw / decompressed / logical SHA、decoder file / mapping SHA、主要manifest SHAもKaggle summaryと一致。exp279 absolute-unary 10.035987よりは改善したがanchorは更新せず、offset grid/process/rate/likelihood救済、PF、blend、selector、raw-test inference、submitへ進まない。独立したprefix-masked offset readoutと、既存のtarget-free future-evidence回復監査を優先する。

- 2026-07-19 に backlog `longtail_prediction_zone_self_gr_loop_closure_readout` / `exp282_longtail_prediction_zone_self_gr_loop_closure_readout` のKaggle private CPU version 1（id_no `127838798`）を0 boosterで完了し、negative resultとして閉じた。donorをsame-well prediction zoneの`0 <= md_since < 500 ft`、receiverを`md_since >= 1000 ft`に固定し、known `TVT_input` prefixを除外して、rolling mean 5、half-window `[8,15,25]`、stride 3、forward/reverse NCCの997,733 target-free edgesを凍結した。edge / finite-score coverageは1.0、truth-before-freezeは0でtechnical guardは全PASS。high-confidence within10は0.554309 vs shuffled 0.551052、lift +0.003257だったが固定precision 0.60に未達、positive liftは4/5 foldsでfold 0が負だった。hidden-like spatial / typewell-purged liftは+0.005491 / +0.004300と正だったが、matched donor-transferはbaseline 8.954770 ftから15.849509 ftへ悪化し、gain -6.894739 ft、改善0/5 folds。frozen edge content SHA `2b9ecbb9...4c28`とraw/decompressed SHAを含む全artifact SHAを取得outputで照合した。1 readout variant / LightGBM config 0 / trained fold 0 / booster 0 / HMM/PF 0、inference/submission disabledを維持し、同一readout内のwindow/stride/confidence/donor範囲の救済grid、soft correction、HMM/PF接続、raw-test inference、submitへ進まない。exp281は独立した並行仮説のままで、本結果によるroute anchor更新はない。

- 2026-07-19 に `self_gr_topk_alternative_mode_branch_future_evidence_readout` / `exp283_self_gr_topk_alternative_mode_branch_future_evidence_readout` のKaggle private CPU version 2（id_no `127849798`）を0 boosterで完了し、negative diagnosticとして閉じた。3,783,989 rows / 773 wells、4,397 events / 13,191 proposals / 103,624 evidence rows、runtime 1,331.408秒。identity / finite coverage / 5-fold coverage / 3段freeze / truth-before-freeze=0などtechnical guardは全PASSした。top-3 within10は0.755288 vs shuffled 0.722083、lift `+0.033204`かつ5/5 folds正、branch-choice AUCもpooled 0.622168・fold min 0.605266で5/5 PASSし、self-GR proposalに弱いsignalは確認した。一方、future-evidence選択はbase RMSE 8.221613から14.606586へ`-6.384973 ft`悪化し、nonregressing fold 0/5、base unique-best false switch 55.5647%、hidden-like 2面も`-7.174194 / -7.125766 ft`。768 wells中637 wellsが悪化し、worstは`-48.601538 ft`だった。metric CSV 8件と5 gzip生成物のSHAを取得outputで照合済み。K/window/horizon/veto/margin/threshold救済、decoder接続、inference、submitをclosedとする。`exp284_prefix_masked_wrong_mode_branch_recovery_backtest`は別の明示overrideでstandalone実行されたが、exp283からscientific promotionは付与しないため、`triggered_fixed_horizon_self_gr_multibranch_hmm_recovery`はexp284結果にかかわらず閉鎖する。

- 2026-07-19 に `exp264_exp263_candidate_confidence_dual_selector` の修正版Stage D Kaggle T4 version 3を完了した。training-only formation 12 selector特徴を除いたStage C v6の88列nested selectorと、非fold-safe 107列を落としたclean 273 downstreamを固定し、control 15 + compact 74列add-only 15 = 30/30 GPU boostersを完走。control `lgb_mean` 10.476169に対しadd-only 8.460811、delta -2.015358、5/5 folds改善。near / mid / 1000+は-0.445903 / -0.756786 / -2.233208、hidden-like spatial / typewell-purgedは-3.073014 / -3.091639で改善した。一方、773 well中255が悪化し、worst `70925e23`は+14.482873で事前上限+0.25を超え、総合guardはFAIL。compact 74列はadd-only gainの76.9258%を占め、上位4 top1-minus-anchorが61.0343%、`beam_mean`予測誤差scoreが5.8196%。後続のユーザー明示overrideでcorrected inference v4（88→74、clean 273 + compact 74 = 347、15 saved TVT、0 training booster）を完了し、submit-check PASSのref `54818932`はPublic LB 7.562。直前ML anchor exp274 7.715を-0.153改善して新ML LB anchorとし、別routeのexp082 ensemble 7.601も-0.039で上回った。同一runの自動ref `54818883`も7.562。hard selector、Viterbi、softmax TVT平均は不採用、worst-well guard FAILも維持し、train-side採用とは分離する。PF/HMM/Beam候補は補助meta featureで最終予測はdownstream LightGBMが生成するため、routeは`ml_model`へ修正した。

- 2026-07-19 に backlog `exp226_prefix_masked_offset_predictability_readout` / `exp285_exp226_prefix_masked_offset_predictability_readout` のKaggle private CPU version 2（id_no `127855223`）を0 boosterで完了し、negative resultとして閉じた。exp226保存5 folds / fold外donor / saved kappaを固定し、known prefix末尾640行をmaskしてpseudo cutからwell末尾までgeometry-only pathを再生、pseudo path SHA後にmasked `TVT_input`、prefix summary SHA後にofficial suffix truthを接続した。version 1はraw horizontalにない`id`列を要求したinput-schema failureで、監査用IDをwell名+row indexから決定的に生成するだけの修正後、766 eligible / 7 ineligible wellsを77.492秒で完走。donor exclusion、mask identity、finite coverage、2段freeze、5 foldsなどtechnical guardは全PASSした。しかしprefix offset median対official full-suffix offset medianはSpearman `-0.004135`、fold `0.055222 / 0.036900 / -0.091083 / -0.033034 / -0.026692`、balanced sign `0.488567`、256 permutation p `0.599222`でprimary guardを全FAIL。slope / driftも`-0.009074 / -0.013928`、1000+は`-0.006022`だった。H256 / nearには`0.186915 / 0.189776`の弱い正相関があるがH512 `0.153020`、H640 `0.131063`へ減衰しfull suffixで消れるため、known-prefix offsetのglobal correction根拠にはしない。exp280の局所likelihood separabilityはlong-horizon offset persistenceを意味せず、exp281 always-on decoderのblend/selector救済、cut/mask/block/summary/threshold grid、prefix-calibrated correction、current-test inference、submitへ進めない。後続exp284も固定horizon self-GR recovery / safety guardをFAILしたため、短距離signalを根拠に新規backlogは追加しない。

- 2026-07-19 に backlog `prefix_masked_wrong_mode_branch_recovery_backtest` / `exp284_prefix_masked_wrong_mode_branch_recovery_backtest` のKaggle private CPU version 2（id_no `127852894`）を0 boosterで完了し、standalone negative diagnosticとして閉じた。exp283生成物へのruntime依存なしで、known prefix末尾640行をmaskし、visible GR likelihoodが支持する`|shift|>=10 ft` wrong modeを注入、safe base + wrong + real/shuffled self-GR top-3をH128/256/512 evidenceで固定比較した。version 1はraw horizontalにない`id`列を要求して評価前に停止し、監査専用IDをwell名+row indexから生成するだけの修正後、766 eligible / 7 ineligible wells、5 foldsを11,717.244秒で完走。mask / branch / evidence finite coverage、fixed branch identity、5 folds、truth-before-freeze=0などtechnical guardは全PASSした。H256 fullはwrong-only 37.557085から26.072230へ`+11.484854 ft`回復し5/5 folds改善したが、safe+wrong pair 23.633930より`-2.438300 ft`悪く0/5 folds改善。pooled pair AUCは0.675153でもfold 3/4が0.509459 / 0.555936、choice accuracyは0.590078。real fullはshuffled 25.520057にも負け、no-injection false switchは30.1724%、H512 gainもH256を下回った。小規模metrics/manifest SHAはKaggle summaryと全件一致。safe base保持はdeliberate wrong-onlyから保護するが、self-GR top-3のincremental value / safetyは否定された。exp283/284/285のself-GR救済としてのparameter変更、triggered decoder、current-test生成、inference、submitは閉じる。2026-07-19のユーザー提案によるType Well GR local modes全件保持は、self-GRを完全除外しhighest-eligible fallbackも使わない別仮説としてexp291へ切り出した。

- 2026-07-20 に backlog `geop_hmm_sparse_addonly_candidate_on_exp264` / `exp286_geop_hmm_sparse_addonly_candidate_on_exp264` をStage Dまで完了した。Stage 0では修正版exp264の12候補へexp279 `geop_hmm`を加えたfull unionがrow / H512 / whole-well oracleを全5 foldsで改善したが、固定top-25% gateのgain保持率`27.710961% < 50%`でsparse gateだけを閉じた。ユーザー明示指示により`geop_hmm`をID one-hot、full availability、`sigma_tvt / source_loglik / loglik_per_row / finite / valid`付きの正式な13番目primitiveとして両legal domainへ追加。Stage B hard selectorはparent12 `8.587004 -> 8.477740`、delta `-0.109265 ft`、3/5 foldsでselector-addition guard PASS。Stage Cは40 CPU models / 25 partitions / compact 77列を生成し、nested hard selector `8.652532 -> 8.448682`、delta `-0.203850 ft`、4/5 folds、score/leakage guard PASS。Stage D T4 version 1（id_no `127886849`）はclean273 + compact77 = 350列の15/15 boostersを完走し、parent12 add-only `8.460811 -> 8.403784`、delta `-0.057027 ft`へpooled改善。near / mid / 1000+とhidden-like 2面も改善した。一方、改善foldは2/5、373 wells改善・400 wells悪化、worst `2d35f86d`は`+5.862833 ft`で、fold数とworst-well条件により総合guard FAIL。平均改善は保持するがtrain-side promotionはせず、inference/submissionは0のまま閉じる。fixed fallback `8.238332`はStage B hard診断でありStage Dの直接比較には使わない。

- 2026-07-19 に backlog `fault_aware_transductive_geological_potential` / `exp289_fault_aware_transductive_geological_potential` のKaggle private CPU Stage 0 version 3（id_no `127879234`）を0 boosterで完了し、scientific guard FAILとしてbranchを閉じた。outer-train `ANCC`とouter-valid `MD/X/Y/Z/TVT_input`だけからcross-well k=12 donor spread、trajectory jump、known-prefix misfitを固定riskへ集約し、773 wells / 320,991 nodesのprimary `suffix_fault_risk_p90`をSHA freeze後にだけexp226 biasへ接続した。technical guardは全PASS、AUC方向とSpearman方向は5/5 foldsで正だったが、`abs(exp226 bias)>=10` AUCは`0.570652 < 0.65`、pooled Spearmanは`0.127885 < 0.25`で総合FAIL。runtime 241.548秒、peak RSS 693.191 MB。v1は全行欠損source ANCC、v2はtarget concat attrsのtechnical errorで停止し、fold-safe修正後のv3で完走した。graph / node / well frozen SHA、7生成物manifest、formation identity、pushed config/source SHAを取得outputで照合済み。局所geological inconsistencyには弱いfold-stable signalがあるが、exp226 rare whole-well biasを説明する強さはない。同一OOFでのedge threshold、formation面、risk aggregation救済、Stage 1/2、inference、submissionは行わない。未着手表から削除し、全体優先は既存LB anchorのtailを0 boosterで再監査するexp276、物理routeはfault救済ではなくknown-prefix内で直接識別性を見るexp290を上位にする。

- 2026-07-19 に backlog `piecewise_datum_physical_smoother` / `exp290_piecewise_datum_physical_smoother` のKaggle private CPU Stage 0 version 1（id_no `127881061`）を0 boosterで完了し、scientific guard FAILとしてbranchを閉じた。exp226 fold-safe geometryをknown prefix末尾から`512/256/128 rows`戻した3 cutで再生し、直後128-row、773 wells / 296,832 rowsをabsolute `[-15,+15] ft`の61 state × 5 duration phase posterior meanで評価した。technical guardは全PASSし、RMSEは1.436926から1.403407へ全5 foldsで改善したが、改善幅は`0.033519 < 0.20 ft`、large-error correction signは`0.483111 < 0.58`、well RMSE p95は`2.437183 -> 2.440010`と微悪化して3 scientific guardをFAILした。runtime 587.042秒、peak RSS 1,215.824 MB。2,319 windowのtruth-after-freeze、finite/coverage/bound、prediction raw/decompressed SHA、state/hyperprior/pseudocut/neighbor SHAを取得outputで照合済み。128-row Stage 0ではminimum duration 256 rowsによりreset probabilityは0で、bounded constant datumのGR識別性は弱いfold-stable改善に留まった。同一OOFでのgrid、clip、pseudo-cut、group、neighbor、likelihood救済、Stage 1、inference、submissionは行わない。未着手表から削除し、本結果だけを根拠とする新しい救済backlogは追加しない。

- 2026-07-19 に `multi_scale_initial_rate_candidates` / `exp268_multi_scale_initial_rate_candidates` のKaggle CPU shard 0/1とaggregate version 1（aggregate id_no `127887734`）を完了した。773 wells / 3,783,989 rowsとshard/aggregate SHAをhard guardし、best direct rate candidate w128はtail30 RMSE `11.938287 -> 11.895581`（`-0.042706 ft`）、initial-rate-5 bankのoracle gainはrow `0.102358 ft`、H256 block `0.102151 ft`、whole-well `0.097314 ft`だった。一方423/773 wellsはrate spread 0で、pairwise path duplicate率は58.99%から88.36%。oracle prediction、mean、selector、inference、submissionは生成していない。子実験exp292のFAIL-closeを受け、rate-window bankをdeployable候補へ昇格せず、train待ち表から削除した。

- 2026-07-19 に `typewell_gr_warp_rate_identifiability_audit` / `exp292_typewell_gr_warp_rate_identifiability_audit` のKaggle private CPU version 1（id_no `127888550`）を0 boosterで完了し、事前登録どおり`FAIL_CLOSE_NO_RESCUE_GRID`としてfrequency-warp rate branchを閉じた。technical guardはPASSしたが、primary H256 eligible coverageは29/773 wells = 3.7516%、row 3.6178%で90% guardをFAIL。candidate-best AUCはreal `0.484190`、shuffle `0.531181`、lift `-0.046991`、正のlift 0/5 folds。全773 wellsでtop1はtail30 safeのまま、RMSE gain 0、改善0/5 foldsだった。1000+ / hidden-like非悪化はsafe完全fallbackによるためpositive evidenceではない。同一truth上のrate/window/horizon/calibration/coverage/weight/threshold救済、top1 replacement、inference、submissionは行わず、未着手表から削除した。

- 2026-07-19 に backlog `calibrated_typewell_gapfill_known_prefix_selfgr_hmm` / `exp294_calibrated_typewell_gapfill_known_prefix_selfgr_hmm` のKaggle private CPU Stage 0 version 1（id_no `127890033`）を0 boosterで完了し、performance hard gate FAILとしてbranchを閉じた。known-prefix raw GR欠損だけをper-well robust-affine calibrated Type Well GRで補完したが、773 wells / 2,319 blocks / 3,865 rowsでcontrol RMSE 8.138531から12.842186へ`+4.703655 ft`悪化し、改善0/5 folds、by-well p95 delta `+15.494311 ft`、157 wells改善 / 610悪化 / 6同値だった。自然欠損run長は全foldでq25/q50/q90 `1/1/3`行のためZNCCは未定義だったが、RMSEとp95だけで棄却は確定する。160.32秒で完走し、observed/raw-mask parity、pseudo-mask fit除外、target fill 0、finite coverage、truth-late-joinなどtechnical gateはPASS。取得した9 artifactのbyte数・raw/decompressed SHAもmanifestと全一致した。Stage 1、affine/window/alpha/threshold救済grid、inference、submissionへ進まず、本結果だけを根拠とする新しい救済backlogも追加しない。

- 2026-07-19 に `prefix_masked_typewell_gr_multimode_safe_beam` / `exp291_prefix_masked_typewell_gr_multimode_safe_beam` のKaggle private CPU version 1（id_no `127882960`）を0 boosterで完了し、scientific / safety guard FAILとしてbranchを閉じた。known prefix末尾640行mask、pre-cut 128行score、H128/H256/H512、固定13 shift bankでself-GRを完全除外したType Well GR local modes全件とsafeを比較した。766 eligible / 7 ineligible wells、5 folds、truth-before-freeze 0、self-GR候補0、candidate/branch/evidence coverage 1.0でtechnical guardは全PASS。一方H256 all-mode RMSEはsafe `4.827483`に対し`22.199818`（gain `-17.372335 ft`、改善0/5 folds）、top1 `18.713110`比も`-3.486709 ft`、matched shuffle `17.360718`にも負けて非悪化0/5 foldsだった。pooled AUCは0.672737だがfold 2/4が0.60未満、balanced accuracy pooled 0.576907、false switch 34.9462%で固定guardをFAILした。runtime 6,805.497秒。K/window/shift/horizon/margin/likelihood/veto救済、decoder、inference、submissionは行わない。未着手表から削除し、本結果だけを根拠とする救済backlogは追加しない。

- 2026-07-20 に backlog `fold_safe_formation_74_addonly_on_exp264` / `exp287_fold_safe_formation_74_addonly_on_exp264` のKaggle private T4 version 5（id_no `127856426`）を完了し、promotion guard FAILとしてtrain-side branchを閉じた。exp218監査でfull-train formation reference依存だった固定74列を、outer-train self-exclusion / outer-valid outer-train-onlyのFormationPlaneKNN / DenseANCCImputerで再生成し、clean 273 + nested compact 74 + formation 74 = 421列を1 variant × 3 configs × 5 folds = 15/15 boosters、control再学習0で評価した。CVはsaved corrected exp264 `8.460811 -> 8.136708`（`-0.324103 ft`）、5/5 folds、near / mid / 1000+、hidden-like 2面をすべて改善した。一方worst well `fb03ae90`は親比`+8.228410 ft`で、clean control比+1/+3/+5 ft悪化well数も`135 -> 140` / `39 -> 40` / `14 -> 19`へ増え、固定well-level safety guardをFAIL。runtime 25,282.477秒。OOF / model / metrics / fold / bucket / hidden / by-well / formation manifest / raw schema audit SHAを選択取得して記録した。guard緩和と同一OOFでのfeature/grid/threshold救済は行わない。その後のユーザー明示指示によりguard FAILを保持したまま保存済みmodel inferenceだけをoverrideし、raw-test current-test生成、40 saved selector + 15 saved TVT model、booster 0でKaggle private CPU inference version 1（id_no `127952811`）を完了した。runtime 448.386秒、14,151 rows、submit-check FAIL/WARN 0。ユーザー完了連絡後に確認した`ref=54842141`はPublic LB 7.530で、exp264 7.562を-0.032改善してML LB anchorを更新し、別routeのexp082 ensemble 7.601も-0.071で上回った。train-side guard FAILとLB anchor更新は分離する。親のPF/HMM/Beam候補は補助compact meta featureで、最終予測はformation add-only downstream LightGBMが生成するためrouteを`ml_model`へ修正した。即時の救済実験は追加せず、train-side adoptionが必要になった場合だけtarget-freeなformation tail属性を読む低優先0-booster attribution案を再検討する。

- 2026-07-20 に backlog `prefix_calibrated_latent_registration_gr_evidence` / `exp297_prefix_calibrated_latent_registration_gr_evidence` のKaggle private CPU version 2（id_no `127897451`）を0 boosterで完了し、事前登録どおり`FAIL_STOP_NO_STAGE4`としてbranchを閉じた。exp293 fixed deployable12、H128/H256/H512 block、21 registration states、prefix Huber affine、MAD scale、Student-t residual/NCC/chain-rule derivative、reliable/unreliable posterior、matched circular shuffleを変更せず、3,783,989 rows / 773 wells / 105,818 block-controlを1,070.800秒で完走した。H256 anchor / oracle RMSE `8.238332 / 3.552829`に対しreal expected RMSEは`8.620041`、headroom recoveryは`-0.116476`で5/5 folds負。shuffle expected RMSE `8.571583`、recovery `-0.101397`にもpooled/5 foldsで負け、1000+とhidden-like 2面もanchorを悪化させた。H256 eligible-stateありblockは29.5044%、real reliable probability中央値は0。704/773 wellsのcalibrationはvalidだったため、coverage不足だけでなく利用可能evidenceのcandidate順位付けも支持されない。truth accessはfreeze前0/後773、target-free/readout 12 SHAは取得outputで全一致し、selected/corrected TVT predictionとsubmissionは生成していない。同一truth上のregistration/component/weight/prior/threshold救済、posterior直接補正、Stage 3、Stage 4、inference、submitへ進まず、未着手表から削除する。新しいexp297救済backlogは追加せず、physical routeの最優先は独立設計済みのexp298 local-shape source監査とexp295 candidate-free SSMのまま維持する。

- 2026-07-20 に backlog `exp226_blockwise_offset_slope_quotient_local_shape_audit` / `exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit` のKaggle private CPU version 2（id_no `127956072`）を0 boosterで完了し、technical PASS / scientific FAILとしてlocal/global decomposition枝を閉じた。exp226 `P_preU=tvt_geop+gr_delta`とexp293 deployable12をtruth-free SHA freezeし、H128/H256/H512/whole-wellのblockwise offset/affine quotientを3,783,989 rows / 773 wellsで監査した。ユーザー承認済みsingleton契約により最終block長1を`4/2/2/0 blocks`だけ全候補共通でaffine分母から除外し、eligible coverage 1.0、長さ2以上invalid 0、bank/block SHA、allowlist、truth-before-freeze 0、alias parityを含むtechnical guardは全PASSした。一方`P_preU`はH256 `0.348268` / rank 4、H512 `0.722409` / rank 5で、post-U `0.304120 / 0.609647`より両方悪化。fold top3は0/5・0/5、H512の1000+とhidden-like 2面もすべて5位だった。strict unique-best比率`0.113630 / 0.115478`だけは通過したが残りscientific guardは全FAIL。version 1は生成物保存後の表示キーだけが`KeyError`となり、監査契約を変えず同じkernel IDのversion 2で主要値・SHA一致を確認してCOMPLETEした。Stage 2/3/4、component/horizon/quotient/scope/平滑化/weight救済、inference、submissionへ進めず、本結果だけを根拠とする新規救済backlogも追加しない。exp293/exp297 fixed12とexp295独立SSMは変更しない。

- 2026-07-20 に backlog `exp223_self_gr_known_tvt_support_gate` / `exp296_exp223_self_gr_known_tvt_support_gate` のKaggle private CPU version 3（id_no `127897387`）を完了し、performance guard FAILとしてstrict support-gate branchを閉じた。exp223 `hmm_selfgr_boost_only_a070_c100`を完全固定し、candidate stateがvisible-prefix finite `TVT_input`のinclusive `[known_tvt_min, known_tvt_max]`外の場合だけself-GR boostをexact 0にした。3,783,989 rows / 773 wells、1 variant / 773 HMM well-runs / LightGBM config・trained fold・booster `0/0/0`、parent control再実行0を16,667.265秒で完走し、outside contribution 0、inside parity 0、truth-before-freeze 0を含むtechnical guard 12/12はPASSした。一方RMSEはsaved exp223 `11.349943 -> 12.159749`、delta `+0.809806 ft`、改善1/5 folds。true-TVT-inside-known-rangeは`-0.571802 ft`改善したがoutsideは`+2.341425 ft`、1000+とhidden-like 2面は`+0.897491 / +1.110813 / +1.118634 ft`悪化した。302 wells改善 / 471悪化、by-well p95 `+1.728087 ft`、worst `2364716c`は`+39.687791 ft`でperformanceは2/10だけPASSした。小規模13 artifactの記録SHAを取得outputと全件照合済み。candidate stateがknown range外という理由だけでsame-well motif evidenceをhard zero化する仮説を棄却し、padding、hole-aware/soft gate、alpha/clip/window/top-k/threshold救済、inference、submissionへ進めない。当初はfeature-only案へ診断証拠だけを引き継いだが、後続のユーザー明示指示により、inside正boostを残した相対boundary priorを原因分離する別仮説として、base-only posterior handoffとsupport-mass-preserving conditional likelihoodを一体化したexp299を設計した。exp299はexp296のpadding/soft-support/threshold救済gridではない。

- 2026-07-21 に `base_posterior_self_gr_boundary_handoff` / `exp299_base_posterior_self_gr_boundary_handoff` のKaggle private CPU version 2（id_no `127957958`）を完了し、train-side guard FAILでbranchを閉じた。version 1のexp209 float32 CSV parity比較bugだけを修正し、同じ1 variant、Pass A/B各773 wells、合計1,546 HMM well-runs、0 boosterを`22,481.454 sec`で完走した。exp209 parityは3,783,989 rows / 773 wellsでmax/mean abs `0/0 ft`、outside contribution 0、boundary neutral 0、truth/control-before-freeze 0をPASSした。technicalはrow gate max `1.0000000000000029`の`2.9e-15`丸め超過だけがFAILして24/25。一方RMSEはsaved exp223 `11.349943 -> 11.789578`、delta `+0.439635 ft`、改善0/5 folds、performance 2/11 PASS。exp296からは`-0.370172 ft`回復したが、inside/outside `+0.415019 / +0.478310 ft`、upper-boundary 0--12 `+0.971110 ft`、1000+ `+0.508852 ft`、p95/worst `+1.454562 / +35.990274 ft`で、hidden-like 2面の約`-0.014 ft`改善を相殺した。微小technical超過を許容してもperformance判定は変わらないため、outside exact-zeroを維持するbase-posterior handoff + conditional normalizationの一体policyを棄却する。handoff/fade/normalizer/alpha/clip/support/threshold救済、version 3、inference、submissionへ進めず、本結果だけを根拠とする新規backlogも追加しない。

- 2026-07-20 に `exp300_exp264_vs_exp274_well_selector_readout` をlocal CPUの保存済みOOF診断として完了した。前のsubmitted ML anchor exp274 raw CatBoost `8.183504`に対し、corrected exp264 Stage D v3は`8.460811`で`+0.277308 ft`、387/773 wellsが悪化し、`>1 / >3 / >5 ft`は`194 / 73 / 40 wells`。outer fold一致710 wellsでも`+0.279430 ft`で、63-well fold差が主因ではなかった。0--1000 ftは全bucket改善し、1000+だけ`+0.327833 ft`、relative-tail q9/q10は`+0.598962 / +0.710069 ft`。target-free raw well特徴の`>3 ft`補助AUCは`0.495675`で単純事前routerを支持せず、exp264/exp274 prediction disagreement RMSEはposthoc AUC `0.946888`だった。corrected Stage C v6 selectorでは`>3 ft`群のerror-margin中央値`0.480424 vs 0.200106`、switches/1000は`35.846954 vs 53.003510`で、低confidenceな迷いではなく高confidence・低switch regimeが特徴。well dominant候補prevalence liftはBeam `3.20x`、Self-GR/LikPF `3.07x`、LikPF `2.31x`。candidate-long全件のoracle分解では`>3 ft`群のexp274 / oracle candidate / selected hard / Stage D RMSEが`10.7964 / 5.2626 / 17.1663 / 16.6532`、MSE項がoracle-vs-exp274 `-88.8676`、selection regret `+266.9864`、Stage D-vs-selected `-17.3527`、final-vs-exp274 `+160.7660`だった。67/73 wellsでselected hardが既にexp274より悪く、主因は良い候補が存在するのにselectorが誤rankingしたこと。regretの52.3%はoracle K16を別候補へ誤rankingしたrow。最大pairのselected Self-GR/LikPF vs oracle K16は悪化群13,331 rowsでRMSE `36.891 vs 9.363 ft`。Beam誤選択は悪化群18,947 rows、全row率`5.256% vs 2.523%`で`2.083x`、母数調整excess約9,851 rowsだった。Stage Dは集約上緩和したが41 wellsで緩和、32 wellsで追加悪化し、後者のうち6 wellsはStage D単独failureだった。時系列switch±5行は正のSSE悪化の14.26%だけで、一律switch suppressionを支持しない。oracle routing、candidate除外・hard fallback・threshold gridを承認せず、新規backlogは増やさず事前固定済みtarget-free risk familyを監査する既存exp276を優先する。

- 2026-07-20 に `exact_hmm_posterior_mode_candidate_audit` / `exp270_exact_hmm_posterior_mode_candidate_audit` の決定論的2 shard version 4とSHA固定aggregate version 4（aggregate id_no `127594551`）を完了し、direct mode-candidate仮説をnegative resultとして閉じた。exp209 raw exact HMMを科学的親に固定し、marginal MAP、global Viterbi、joint exact top-5をposterior meanと同じHMM passから生成した。363 / 410 well shard、合計3,783,989 rows / 773 wells、exp209 parity max 0.0 ft、ID/order/finite/禁止列、全artifact SHA、candidate raw/decompressed SHA、prediction content SHAはすべてPASS。aggregateは156.241秒、peak RSS 3,097.277 MB。direct RMSEはposterior mean `11.938287`が最良で、marginal MAP `12.592479`（`+0.654192 ft`）、global Viterbi `15.551665`（`+3.613377 ft`）、top-2からtop-5も全悪化した。hidden-like spatial / typewell-purgedもposterior mean `12.564491 / 12.367244`が最良。all-mode oracleはrow `7.516850`、block128 `7.567530`、well `8.536362`だがtrue TVTを使う診断専用であり、mean / MAP / Viterbiだけとの差は最大`0.000342 ft`のためtop-2からtop-5の追加価値はほぼない。oracle prediction、selector、inference、submissionは生成していない。実装待ち表から削除し、oracleだけを根拠とする救済backlogは追加せず、既存高優先実験を維持する。

- 2026-07-20 に `gauge_invariant_multiformation_edge_potential` / `exp301_gauge_invariant_multiformation_edge_potential` のKaggle private CPU version 2（id_no `128007163`）を完了し、Stage 0 technical negativeとしてbranchを閉じた。version 1はfiltered edge DataFrameの代わりにboolean Seriesを渡す1行型バグでpre-solver ERRORとなり、同一contractのversion 2で修復した。6 formation edge identityは最大RMSE `0.008133 ft`、median6最大`0.007870 ft`、eligible edge fraction 1.0、row/fold/well identity、bilinear basis、leakage、runtime guardは全PASS。一方、query component donor coverageはfold `0.986729 / 0.979238 / 0.979066 / 0.969525 / 0.995853`（pooled `0.982164`、全query geometry中90,827 rows unsupported）、active component donor coverageは`0.96 / 0.92 / 0.92 / 0.96 / 0.98`でexact 1.0を満たさなかった。事前policyどおりsolver fit 0でStage 1、OOF、direct RMSE、candidate noveltyを実行せず、inference、submission、案2/案3へ進まない。同一OOFでgrid spacing、halo、adjacencyを調整する救済は行わず、再訪は下記の低優先truth-free component connectivity readoutを先に通す場合だけとする。

- 2026-07-21 に `typewell_group_prefix_suffix_gr_calibration_readout` / `exp311_typewell_group_prefix_suffix_gr_calibration_readout` のKaggle private CPU version 1（id_no `128085784`）を完了した。1 diagnostic / 5 folds / model・booster・decoder 0、runtime `246.631 sec`。primary `native_overlap_1` same-group held-out-wellは760/773 wellsで利用可能、identity GR-RMSE `11.745716`から`11.369495`へ`0.376220`改善し、5/5 folds、group-shuffle差`0.240055`、noise R² `0.202320`、late-truth境界をPASSした。一方fit-RMSE R²は`-0.003255`で閾値`0.20`を満たさず、worst-well deltaは`+12.914716`で上限`+0.25`を大幅に超えた。spatial/typewell-purgedもpooled gain `0.349090`はあるがworst `+12.578262`、exact-hashは利用可能28/773 wells・gain `0.013695`に留まった。群noiseの平均的転送性だけではfit品質とtail safetyを保証できないため、同一OOFでの閾値・shrinkage・group定義救済は行わずbranchを閉じる。exp312〜320は当初停止したが、後にユーザー判断でexp312だけを明示上書きして実行した。次の優先順位は独立したP0のexp321 Stage A/Bと、exp304 PASSから続くexp305を維持する。

- 2026-07-21、上記exp311の固定gate FAILを確認したうえで、ユーザーは平均GR gain `0.376220`と5/5 folds改善を根拠にexp312だけを明示上書きし、exp293 deployable12固定の`exp312_typewell_group_conditional_gr_emission_table`を実装・実行した。Kaggle private CPU version 1（id_no `128090149`）、scientific 1 + controls 2 / 5 folds / model・booster・decoder 0、runtime `326.623 sec`。global-unconditional baselineに対し、group × Type Well GR decile × |gradient| tertile × missing flagのdf=5、k=200 conditional tableはMRR `0.336112 → 0.334519`（`-0.001592`）、top3 `-0.002444`、改善`0/5` folds、group-shuffle差`+0.001611`、hidden-like 2面FAILだった。fallback率`1.823%`と全foldのlate-truth境界はPASSし、candidate-TVT shift差は`+0.063809`だったが、群条件づけの追加価値を支持しない。予定10生成物とraw SHA 9/9一致を確認し、candidate生成/model/decoder/inference/submissionは0。bin/df/kの救済を行わずbranchを閉じ、exp313〜320の停止を維持する。同系救済backlogは追加せず、優先順位は独立したexp321とexp305を維持する。

- 2026-07-21 に `gr_denoiser_emission_separability_readout` / `exp304_gr_denoiser_emission_separability_readout` のKaggle private CPU version 1（id_no `128011752`）を0 boosterで完了し、technical/quality gateをPASSした。3,783,989 rows / 773 wells / 7,787 blocks / 13 shifts、runtime `4,740.758 sec`。raw MRR/top3 `0.389626 / 0.452421`に対し、stationary db4 level-3 SWTは`0.424724 / 0.504687`で`+0.035098 / +0.052267`改善し、MRR/top3とも5/5 folds、real-vs-shuffled 5/5 folds、1000+、hidden-like 2面、sharp-edge、top1、decoy-gapの全事前gateを通過した。raw/SWTは全1,546 series technical PASS、silent fallback 0。一方robust RTSは1,531 failures、L1 trendは974 failuresでtechnical FAILとし、反復・閾値の救済gridは行わない。`selected_denoiser=swt_db4_l3`を固定し、実装済みbacklogからexp304を削除する。予約案2のfixed beta 0.15 tempered raw/SWT exact-HMMだけを下記の高優先別exp候補へ開き、SWT選択のためRTS variance専用案3は閉じ、案4は案2 PASSまで開始しない。

- 2026-07-20 に `exp226_multiscale_k_segment_candidate_audit` / `exp302_exp226_multiscale_k_segment_candidate_audit` のKaggle private CPU version 2（id_no `128010921`）を完了した。K12/K24だけをexp226固定5 foldsで生成し、2 variants × 5 folds = 10 CPU runs、0 booster、K16 control再生成0を`1281.068 sec`で完走した。3,783,989 rows / 773 wells、truth-before-freeze 0、candidate-bank/block/control SHA、finite coverageを含むtechnical guardは全PASS。directはK12 `9.551938`（K16比`+0.124828 ft`、0/5 folds）、K24 `9.413244`（`-0.013865 ft`、3/5 folds）で両方FAILし、direct候補へ昇格させない。一方、exp293 fixed12へのadd-one H512 / whole-well oracle改善はK12 `+0.066095 / +0.068466 ft`、K24 `+0.083901 / +0.066231 ft`、strict unique-best `10.6973% / 10.8899%`、両方5/5 foldsでcandidate novelty guardをPASSした。output manifest 16/16件とK12/K24/block decompressed SHAを照合済み。完了済みexp302をbacklogから削除し、exp302側のexp303 dependencyだけを充足済みに更新する。その後exp276 corrected-parent version 3も固定guard FAILで完了し、exp303の全dependencyを成立させ、K値/weight/selector救済、inference、submissionは行わない。

- 2026-07-22、`exp334_equal_well_loss_weighting_on_exp287` のKaggle T4 train version 2（id_no `128110184`）が15/15 boostersを`21882.805369142 sec`で完了した。CVは`8.09349752413077`、exp287比`-0.04321069622868201 ft`、5/5 folds改善。near/mid/1000+と2 hidden-like scopeも全gate内だった。一方、by-well p95 deltaは`+0.429584617 ft`、exp264比worst-wellは`+7.156485377 ft`、`+1/+3/+5 ft`悪化well数は`133/40/19`で、p95、worst、`+3/+5`がFAIL。well均等lossはglobalとtailの一部を改善したがsevere tailを回復できず、固定AND gate不通過として非昇格で閉じた。追加train、weight grid、inference、submissionは実行しない。OOF 3,783,989行、773 wells、15 models、非model成果物11件を実ファイルとmanifest SHAで監査し全件一致した。この結果により、exp334 FAILまたはtail改善不十分時だけ再開するとしていた0-booster formation tail attribution readoutの再開条件は成立した。
- 2026-07-22、再開条件が成立した0-booster候補を`exp336_exp287_formation_tail_attribution_readout`として実装し、Kaggle private CPU version 2（id_no `128221753`）をreadout本体`92.458 sec`で完了した。exp287 outer-valid formation cache 5 partition / 3,783,989 rows / 773 wellsからplane距離、dense距離、dense不確実性、plane-dense不一致、formation spread、known-prefix calibration errorの6 familyをtruth/errorなしでfreezeし、manifest SHA `e65a9924...49f`を検証後だけcorrected-exp264 OOFをjoinした。全familyでstrict edge、error非依存、global/fold/hidden-like coverageを通過したが、固定AND gateのpassed familyは`0/6`。最も近いdense距離はglobal Q4-Q1 mean`+0.350471 ft`、median正、5/5 folds正でもhidden-like 2面が`-0.019368/-0.037255 ft`と逆方向。formation spreadとknown-prefix calibrationはhidden-like 2面を通したがglobal effect不足だった。version 1は同一SHA assignmentの3 path複製を非一意としたtechnical ERRORで、scientific contract不変のdeterministic resolverだけを修正した。11 artifactの存在とreproducibility manifest記録10子artifactのSHA一致を確認した。`NO_STABLE_FORMATION_ATTRIBUTION_CLOSE`として未着手表から削除し、同じOOFでのfamily/threshold/weight/clip/shrink/gate救済、別介入、inference、submissionへ進めない。model/config/fold/booster/control再学習は`0/0/0/0/0`。

- 2026-07-22、`prefix_backtested_structure_sigma_gr` / `exp337_prefix_backtested_structure_sigma_gr`のStage 0をimplementation-onlyで実装した。known-prefix GR residualをorigin `60%/80%`以前だけでfreezeし、内部60/40のearly population stdとlate zero-center MSEから`tau_structure^2=max(0,MSE_late-sigma_early^2)`を計算する。直後20% finite residualのGaussian NLLでfinite-only、exp209 zero-fill、structure-addedを比較し、coverage/fallback、pooled/4-of-5 folds、zero-fill比`0.005/residual`、full-prefix median tau、lower clipを固定AND gateにした。total finite 50、early/late各20未満は同prefixのzero-fill scaleへno-op fallbackする。compact self-contained train/inference候補と専用testを作成し、`10 passed`、Jupytext/py_compile/RuffをPASS。Stage 0は1 diagnostic、HMM/model/booster/control再実行はすべて0。正規Notebook上書き、Kaggle package/push/run、Stage 1、inference、submissionは行っていない。これにより未着手バックログから実装済み・Kaggle train待ちへ移し、優先度は低-中・P2のまま維持する。
- 2026-07-22、`exp337_prefix_backtested_structure_sigma_gr`のKaggle private CPU Stage 0 version 1（id_no `128220965`）を`143.899363 sec`、1 diagnostic / HMM・model・LightGBM・PF・Beam・booster・control再実行各0で完了した。両originとも773/773 wellsを評価しfallback 0。structure-addedはzero-fillに5/5 foldsで勝ち、NLL gainはorigin 0.60/0.80で`0.515373 / 0.556105` per residualだったが、finite-onlyには両originとも0/5 foldsで負けた。pooled NLLもorigin 0.60で`3.027165 → 3.073866`、0.80で`2.971854 → 3.015784`と悪化し、full-prefix median `tau_structure=0.0`も固定gate `>=5.0`をFAILした。追加構造分散仮説は支持されないと判断し、split/threshold/scale/likelihood救済、Stage 1 HMM、inference、submissionなしで枝を閉じる。完了済みexp337を実装済みbacklogから削除し、既存候補の優先順位は変更しない。

- 2026-07-22、`exp332_prefix_gr_unary_fixed_window_structured_ssm`のKaggle T4 Stage 0 version 1（id_no `128231704`）を固定16 windows / temporary neural model 1 / 永続model・LightGBM・booster・PF/Beam・control再学習各0で完了した。measurement 112行はstructured train 16、forward-only 16、full-well unary 32、exact decode 48で全時間が正。p50/保守的fold外挿は`12.744536 / 13.151137 h`、peak memoryは`1.203263 GB`。memoryはPASSしたがruntimeは`8.5 h`上限をFAILし、主因はfit structured training `9.214264 h`と3-control decode `2.937457 h`だった。outer-valid truth accessとStage A modelは各0、selection/boundary/measurement SHAはreportと実ファイルで一致した。`close_without_window_or_loss_rescue`としてbranchを閉じ、完了済みexp332を実装済みbacklogから削除する。同系救済を追加せず既存独立候補の優先順位を維持する。

- 2026-07-22、ユーザーの明示依頼により、terminal closedのexp332を再開せず、その着眼点を別実験として検証する2案をdesign-onlyで固定した。手堅い先行案`exp347_prefix_gr_unary_batched_window_exact_ssm`は、exp332の`1 window/batch × accumulation 4`を`4 windows/batch × accumulation 1`へ置換する計算実装だけを変更し、objective/window/boundary/architecture/state grammar/control/full-well gateを固定する。Stage 0は1 benchmark variant / 固定16 windows / temporary model 1 / persisted model・trained fold・LightGBM・booster・PF/Beam・control再学習各0で、scalar loss/posterior/gradient/update parity、T4保守的`<=8.5 h`、peak`<=14 GB`、exp332比`>=1.55x`をAND gateとする。高リスク後続`exp348_prefix_gr_unary_window_path_ranking_ssm`は、同じ256-row window着眼点を保ち、exact structured NLLだけをpositive 1対固定negative最大16のmargin`0.05` rankingへ置換する。unique negative`>=12`、positive top-1`>=0.80`、margin`>=0.02`、同じruntime/memory gateをStage 0で要求し、full-well exact decodeの科学gateなしには昇格させない。exp347をexp348より先行し、両者を同時GPU実行しない。2件ともscaffold/config/steeringだけで、実装、Notebook採用、Kaggle package/push/run、推論、提出は未承認・未実施。現行P1のexp335を追い越さず、exp347をP2、exp348をP3とする。

- 2026-07-22、ユーザーの明示依頼により`exp347_prefix_gr_unary_batched_window_exact_ssm`をimplementation-onlyで実装した。exp332 compactの13章科学契約を維持し、row/position/rate/inactive-window mask付き4-window exact forward-backward、per-window normalized loss平均、1 batch/1 AdamW step、scalar/batch loss・partition・posterior・unary gradient・1-step update parity report、fixed16 Stage 0、stable length順4-wellのreal/shuffle/geometry full-well decode、Stage A freeze-first batched decodeを別名compact train候補へ実装した。temporary neural modelは1個だけで、parityはfreeze済みunary tensorをparameterとして比較する。fail-closed inference候補と専用testを作成し、Jupytext/py_compile/Ruff/strict validationをPASS、pytestは`16 passed, 2 skipped`。skipはローカルPyTorch未導入による数値testで、Kaggle T4 Stage 0の必須gateとして残す。正規Notebook上書き、Kaggle package/push/run、Stage A、推論、提出は未実施。未着手表から実装済み・Kaggle train待ちへ移し、P2・exp348より先行を維持する。

- 2026-07-23、`exp347_prefix_gr_unary_batched_window_exact_ssm`のKaggle T4固定16-window Stage 0 version 1（id_no `128239400`）を1 benchmark variant / temporary neural model 1 / 永続model・trained fold・LightGBM・booster・PF/Beam・control/親再学習各0で完了した。4-window batch化はp50/保守的fold外挿`4.741982 / 5.108737 h`、exp332比`2.574244x`、peak`5.928168 GB`で全compute gateをPASSした。scalar/batch loss・partition・AdamW update差は0、gradient差`1.4319085e-8`、padding/finiteもPASSしたが、posterior max abs error`1.4662743e-5`が固定上限`1e-6`をFAILした。outer-valid truth accessとStage A modelは各0。window/boundary/padding/parity/measurement/report/log SHAを実ファイルと照合し、AND gate不通過の`close_without_batch_or_science_rescue`としてbranchを閉じた。exp347を実装済み・train待ち表から削除し、batch/padding/compile/fused kernel/閾値/科学契約の救済を追加しない。独立した高リスクP3のexp348は先行条件を満たしたが、実装・実行は別判断を要するため優先度を上げない。
- 2026-07-24、ユーザーの明示依頼により`exp348_prefix_gr_unary_window_path_ranking_ssm`をimplementation-onlyで実装した。exp332 compactの13章、256-row window、teacher boundary、neural unary、local CE`0.25`、fixed exp209 grammar、full-well exact decodeを維持し、trainingの4 partition sweepだけをpositive 1対固定negative最大16のgather-only rankingへ置換した。positiveはsigma`0.35 ft` label-conditioned joint Viterbi、negativeはposition offset 6 / constant rate offset 4 / midpoint pulse 4 / saved exp209 1 / geometry-only 1。grid外はclipせず除外し、fixed grammarへlegal projection後にdedup、unique`>=12`を要求する。boundary/transition potentialはfit前に事前計算し、window/path SHA・dedup理由とともにfreezeする。Stage 0固定16 windowsは12 optimizer + 4 early holdoutとし、top-1`>=0.80`、mean margin`>=0.02`、path-bank生成込み保守的runtime`<=8.5 h`、peak`<=14 GB`をAND評価する。compact train / fail-closed inference候補と専用testを作成し、`16 passed, 1 skipped`。skipはローカルPyTorch未導入による数値test。正規Notebook上書き、Kaggle package/push/run、Stage A、推論、提出は未実施。1 variant / temporary neural model 1 / persisted model・trained fold・LightGBM・booster・PF/Beam・control再学習各0のStage 0は別承認待ちで、P3を維持する。

- 2026-07-25、`exp348_prefix_gr_unary_window_path_ranking_ssm`のKaggle private T4 Stage 0 version 2（id_no `128524049`）を固定16 windows（optimizer 12 / early holdout 4）、1 benchmark variant / temporary model 1 / 永続model・trained fold・LightGBM・booster・PF/Beam・親/control再学習各0で完了した。version 1はraw CSVに存在しない`id`列を仮定して学習前ERRORとなり、正規ID契約`{well}_{row_index}`へのtechnical fixだけでversion 2へ進んだ。version 2は全window unique negative 16、outer-valid truth access 0、training partition sweep 0、path-bank pre-fit freeze、peak`1.193590 GB`でtechnical/memoryをPASS。一方、early-holdout positive top-1`0.0 < 0.80`、positive-max-negative margin`-0.388485 < 0.02`、14,816 path-bank workload込み保守的fold外挿`75.356700 h > 8.5 h`でlearning/runtimeをFAILした。report、logs、必要な6 artifactとraw/decompressed SHAを照合し、固定AND gate不通過の`close_without_negative_bank_margin_or_science_rescue`としてbranchを閉じた。negative family/count、margin、loss、window、architecture、decoder、epochの救済、Stage A/B/C、推論、提出は行わない。完了済みexp348を実装済み・train待ち表から削除した。

- 2026-07-23、`exp287_u_boundary_continuity_fade` / `exp349_exp287_u_boundary_continuity_fade`のKaggle private CPU version 2（id_no `128239658`）を1 fixed postprocess / 5 reporting folds / model・booster・PF/Beam/HMM・control再学習・GPU各0で完了した。version 1はgap bucket文字列化のpandas返り値型差でfreeze前ERRORとなり、仮説・入力・cap/tau・gateを変えない型互換修正だけで再実行した。3,783,989 rows / 773 wells、parent OOF SHA、全well prefix/suffix、truth-before-freeze 0、formula/cap/fade、candidate SHA readbackを含むtechnical gateは全PASS。CVはexp287 `8.136708220`から`8.135096925`へ`0.001611295 ft`改善し、5/5 folds、0--240 `0.110003778 ft`、hidden-like 2面、far 3帯、by-well median/p95/worstはPASSした。一方、pooled改善は事前下限`0.020 ft`の約8.1%で唯一のscientific FAILとなった。1000+改善が`0.000002 ft`と実質ゼロで、境界近傍の効果が未知suffix全体を動かす量に達しない。`FAIL_CLOSE_NO_RESCUE`として完了済みexp349をbacklogから削除し、cap/tau/threshold/distance/well gate/blend/別親救済、inference、submissionを行わない。continuity再訪は独立したtarget-free add-only feature／selector仮説が得られた場合だけとし、既存候補の優先順位は上げない。

- 2026-07-22、`robust_rts_l1_convergence_calibration_audit` / `exp306_robust_rts_l1_convergence_calibration_audit`のKaggle private CPU Stage 0 version 1（id_no `128231380`）を固定64 wells、3 branches、384 core + 16 parity series-runsで完了した。L1 max2000は`128/128` convergence、min/mean/max iterations `264/656.758/1993`、実測`25.161 sec`、773-well外挿`303.896 sec`、8-well x 2 series exact parityを全PASSし、唯一のfull-eligible branchになった。RTS A=`32,1e-6`は`7/128`、条件付きB=`32,1e-4`は`108/128` convergenceで、Bの残るFAILはhorizontal 7 / typewell 13。両RTSともfinite/order/fallback/runtimeはPASSしたがall-convergence gateを満たさず、追加iterations/tolerance/grid救済なしで不適格として閉じた。input/output/statusのraw/decompressed SHAは取得ファイルとgateで一致し、truth/scientific score、prediction、submissionは未生成。完了済みexp306を実装済み・train待ち表から削除した。2026-07-23のユーザー指定によりL1 773 wells / 1,546 series full technical auditは別実験`exp351_exp306_l1_full_convergence_audit`へdesign-onlyで切り出し、scientific score、RTS救済、inference、submissionへ自動進行しない。exp304 selected SWTとexp305 closed判断は変更しない。
- 2026-07-23、`exp351_exp306_l1_full_convergence_audit`のKaggle private CPU version 1（id_no `128354027`）を固定L1 max2000 1 branch / 773 wells / 1,546 series-runs / model・LightGBM・fold・HMM・PF・Beam・booster・control再実行・GPU各0で完了した。親version 1 artifact、raw identity、coverage、finite/order、fallback/error、64-well input/output/status、8-well output/status/iteration exact SHA parity、runtime `329.250 sec <= 8.5 h`は全PASSしたが、horizontal 9 seriesが全てiteration 2000で未収束となり、convergence/technicalは`1,537/1,546`だった。typewellは`773/773`、horizontalは`764/773`。固定all-series AND gateを満たさないため`full_technical_fail_closed`とし、iteration/tolerance/lambda/rho/adaptive rho/grid救済、scientific score、exp304 selected SWT変更、HMM/PF/Beam、inference、submissionなしで閉じた。完了済みexp351を実装済み・train待ち表から削除し、本結果だけを根拠とする新規L1 solver救済backlogは追加しない。

- 2026-07-24、`exp363_sticky_gr_reliability_exact_hmm`のKaggle private CPU
  Stage 0 version 1（id_no `128370770`）を`497.082523 sec`、diagnostic 1 /
  5 reporting folds / HMM・model・LightGBM・trained fold・booster・parent control
  再実行各0で完了した。3,783,989 rows / 773 wells / 15,174 blocks、
  truth-before-freeze 0、finite、fold、circular offset、strict quartileを満たし
  technical gateはPASS。pooled bad10 AUC`0.607552`、circular差`+0.023556`、
  Q4-Q1 mean block RMSE`+4.816306 ft`、5/5 fold AUC、hidden-like
  typewell-purged AUC`0.552195`はPASSした。一方、hidden-like spatial AUC
  `0.546058 < 0.55`とrow-weighted weak mass`0.589441 > 0.50`が固定AND gateを
  FAILした。fixed qはbad block signalを持つが、一時的な観測不良より広い区間をweakと
  解釈し、spatial stressへ十分に転送しないnegative resultと判断する。decisionは
  `stage_0_failed_close_without_rescue`。transition/multiplier/sigma/block/threshold/
  blend救済、Stage 1、inference、submissionなしでbranchを閉じ、完了済みexp363を
  backlogから削除する。同じq契約に依存するexp368はこの時点ではblocked/demotedとし、次は
  reliability抑制の救済ではなく、観測位置ずれを物理位置から分離する既存の独立案
  exp365を候補として維持する。その後2026-07-25の明示依頼によりexp368の0-PF
  Stage 0だけを実装したが、Kaggle実行とStage 1は未承認であり、exp363 negative dependency
  と低優先判断は維持する。

- 2026-07-25、`exp368_marginalized_reliability_pf`のKaggle private CPU
  Stage 0 version 1（id_no `128591117`）を`630.531264 sec`、diagnostic 1 /
  reporting folds 5 / PF・control replay・model・LightGBM・trained fold・booster・
  親control再実行各0で完了した。3,783,989 rows / 773 wells / 15,174 suffix blocks /
  49,472 known-prefix held-out rows、truth-before-freeze 0、finite、fold、circular
  offset、strict quartileを満たしtechnical gateはPASS。pooled bad10 AUC
  `0.636675`、circular差`+0.058264`、5/5 fold AUC、hidden-like spatial /
  typewell-purged AUC`0.641795 / 0.636115`はPASSした。一方、known-prefix
  predictive NLL gainは`0.037356% < 1%`、row-weighted weak massは
  `0.009689 < 0.02`で固定AND gateをFAILした。suffix errorの識別signalはあるが、
  weak stateは稀にしか発火せず既知prefixの予測効用も不足すると判断する。decisionは
  `stage_0_failed_close_without_rescue`、Stage 1 eligibilityはfalse。
  transition/sigma multiplier/block/threshold/gate/blend救済、再push、Stage 1、
  inference、submissionなしでbranchを閉じ、完了済みexp368をbacklogから削除する。
  同じqの救済は追加せず、原因確認が必要な場合だけ下記の独立truth-free
  prefix/suffix activation shift auditをP4に置く。既存P1/P2候補の優先度は変更しない。

- 2026-07-25、`exp365_bounded_gr_registration_offset_hmm`のKaggle private CPU
  Stage 0 version 2（id_no `128537562`）をdiagnostic 1 / offset states 5 /
  reporting folds 5 / resource wells 16 / exact-HMM well-run・LightGBM config・
  trained fold・booster・parent control再実行・GPU各0で完了した。773 wells /
  18,465 rolling windows / 915,301 observed held-out rows、suffix truth read 0、
  physical prediction 0でtechnical gateは全PASS。real delta=0比NLL gainは
  `+5.430399%`だったが、missing maskと観測値multisetを保持したcircular controlが
  `+15.311425%`、real-minus-circularは`-9.881025%`で全fold負、passing fold
  `0/5`だった。adjacent-window sign agreementも`0.580771 < 0.60`、固定runtime
  projectionも`56,429.34 > 30,600 sec`。version 1のtechnical falseはprobability
  intervalの機械精度判定だけで、`atol=1e-12`へ修正したversion 2のscientific
  contract / input manifest / rolling ledger / posterior / resource projection /
  fold metricsはversion 1とbyte-identicalだった。科学判定を
  `STAGE0_FAIL_CLOSE_WITHOUT_RESCUE`とし、offset/transition/grid、sigma、
  runtime係数、gate、controlの救済、Stage 1、inference、submissionなしで閉じる。
  完了済みexp365を実装済みbacklogから削除し、本結果だけに依存する救済候補は追加しない。

- 2026-07-24、`exp226_formation_conditioned_k16_donor_kernel` /
  `exp376_exp226_formation_conditioned_k16_donor_kernel`のKaggle private CPU
  version 2（id_no `128436621`）を1 variant / 5 reporting folds /
  model・booster・parent control再実行各0で完了し、technical / target-free
  Stage 0 PASS、direct / candidate novelty FAILとしてbranchを閉じた。version 1は
  5/5 folds予測後にlist-valued reference manifestのlogical hashでtruth前ERRORとなり、
  container cellだけをcanonical JSON化する局所修正でversion 2を完走した。
  Stage 0は3,783,989 rows / 773 wells / 12,368 segments、factor
  `0.511501--1.0`、finite 1.0、fallback 0、ESS比p05 `0.927173`、
  valid reference/truth/formation read 0で全PASS。一方direct RMSEはexp226
  `9.427110 -> 9.443257`（`+0.016148 ft`）、改善1/5 folds、by-well p95
  `+0.376679 ft`、worst `a3518960 +1.891560 ft`でFAIL。fixed12へのadd-oneは
  H512 / whole-well改善`0.019404 / 0.015542 ft`で閾値`0.05 ft`未達。
  strict unique-best `9.3874%`と5/5 foldsだけはPASSしたが、exp226との予測相関は
  `0.999999782`で増分が小さい。summary/freeze/SHA/guard/metrics/by-well/input/
  schemaを選択取得してlogical/decompressed SHAを記録した。weight/surface/signature/
  K/bandwidth救済、current-test、selector、inference、submission、version 3は行わず、
  完了済みexp376をtrain待ち表から削除する。exp362のnegative evidenceとexp356の
  support待ちblocked判断を維持し、同じK16 donor条件付けの救済backlogは追加しない。

- 2026-07-24、train-only地層列と正解TVTを物理モデルへ利用する新規branchを
  `exp377--382`として設計確定した。exp376の「参照weight変更」はexp226との相関
  `0.999999782`で新規性不足だったため、同じdonor weight救済ではなく、補間する物理量を
  `d(TVT+Z-F_f)/dMD`へ変える。主経路は0-HMM識別可能性のexp377、7候補の直接・新規性監査
  exp378、Public-LB-best参照exp335へのstrict-nested 20特徴add-only exp382の順とする。
  exp379 exact HMMとexp380 stratified PFはexp378 novelty PASS後の代替物理経路で同時着手せず、
  exp381 contact-order semi-Markov HMMは別仮説として0-HMM gateを先行する。6件とも
  experiment/scaffoldとsteeringだけを作成し、実装、Notebook採用、Kaggle package/push/run、
  GPU学習、PF/HMM full run、inference、submissionは行っていない。

- 2026-07-24、ユーザー指示により上記branchの先頭
  `exp377_formation_relative_k16_slope_identifiability_readout`を実装し、追加の実行指示で
  正規train Notebookを採用してKaggle CPU v1/v2
  （`kentookumura/exp377-formation-relative-k16-slope-readout-train`、id_no
  `128452991`）を完了した。exp226 outer5/K16/方位projection/XY最近傍50/
  bandwidth 500/ridge 1、outer-trainの6 `d(S-F_f)/dMD` field、
  FormationPlaneKNN(k=10)、median6を固定した。3,783,989 rows / 773 wells /
  12,368 segments / 5 folds、valid truth/formation read 0、source-valid overlap 0、
  primary coverage 1.0、surface fallback 0.0はPASSした。v1ではeffective donors p05
  `2.59469484575288 < 10`がStage 0唯一のFAILとなり、truth前に停止した。コード監査で
  direct controlと6 relative fieldが同じeligible donor XYと距離weightを共有するため、
  ユーザー承認済みv2ではK16/kernel/formation式/primary/Stage 1 gateを変えず、
  この数値checkだけをreport-only warningとして保持した。target-free bundle SHAは
  `944af71f245e5e4615953c7d69fbbb3f22e48757cf63d8474e16d0398a683e5a`。
  v2 Stage 0はblocking checksを全PASSしてtruth-late Stage 1へ到達したが、
  segment rate RMSEはdirect `0.012301→0.038454`、累積path RMSEは
  `16.100131→38.776238 ft`（`+22.676107 ft`悪化）、rate/path改善foldはいずれも
  `0/5`、609/773 wells悪化、p95 `+49.434562 ft`、worst `+408.044686 ft`で
  7 checksすべてFAILした。個別6 formation pathも`39.022186--40.355628 ft`で
  directより全悪化したため、median集約だけの問題ではない。decisionは
  `close_and_block_exp378_exp379_exp380_without_surface_kernel_or_scope_rescue`。
  exp378/379/380/382を未実装・未実行のまま閉じ、parameter grid、posthoc surface選択、
  HMM/PF/ML救済、inference、submissionを行わない。独立仮説のexp381とexp383以降の判断は
  この結果だけでは変更しない。

- 2026-07-24、ユーザーの「全train坑井の正解TVTとtrain-only地層列を使い、
  物理モデル単独でPublic LB 6.5を目指す」という目的を受け、exp226の小幅K16派生ではなく
  全TVTから地層ドリフト場を再構築するP0--P2を`exp383--385`として設計確定した。
  P0 `exp383_all_tvt_stratigraphic_vector_drift_field`は全outer-train TVTを
  64/256/1024 ft windowへ展開し、6地層surface、absolute/vector field、全prefix vertical bias、
  uncertaintyによるexp226縮約、banded physical path solveを1候補へ固定する。
  P1 `exp384_fault_aware_piecewise_stratigraphic_vector_field`はexp383が1 ft以上改善した場合だけ、
  outer-train formation/structural discontinuity graphからpiecewise component fieldを作る。
  P2 `exp385_gr_typewell_likelihood_on_vector_drift_paths`はexp383/384がともにPASSした場合だけ、
  base 1 + component最大8の物理pathをtarget horizontal GRとtypewell GRの固定Student-t尤度、
  exact forward-backwardで周辺化する。3件ともroute=`pf_beam`で設計し、この時点では
  design-only、Kaggle package/push/run、inference、submissionは0。
  exp377 v2は上記のscientific FAILで終了し、exp378--380/382も閉鎖した。
  exp383--385はexp377とは異なる全TVT地層ドリフト場branchとして独立に扱う。
  exp383は2026-07-24のユーザー指示でcompact self-contained実装まで完了した。
  その後の実行指示により正規Notebookを採用し、private CPU / internet offの
  canonical packageを監査した。1 candidate / 5 folds / model・HMM・PF・Beam・booster各0、
  exp226再実行0の16-well Stage 0 preflightをversion 1で実行した。
  version 1はtruth join前、fold 0の209,467 donor windowsへのsurface付与後に、
  scaleをjoin keyへ含めなかった`MergeError`で`22,055.465 sec`時点に停止した。
  scale込み一意donor `query_id`へローカル修正し、専用test `15 passed`を確認したが、
  全fold `1,043,436` windowsへのsurface stage投影は`109,866.787 sec`
  （30.52時間）で固定gate`30,600 sec`の3.5904倍だった。
  join修正版の再push、full run、Stage 1、inference、submissionを行わず、
  `stage0_resource_fail_closed`としてexp383を閉じる。必須PASS artifactが得られないため、
  exp384は実装済み未実行、exp385はdesign-only未実行のまま依存失敗で閉じる。
  2026-07-25、ユーザーがexp383とその後続exp384/385の閉鎖を明示確認したため、
  3件を再実行・再開候補には残さない。

- 2026-07-24、ユーザー指示によりP1
  `exp384_fault_aware_piecewise_stratigraphic_vector_field`をimplementation-onlyで先行実装した。
  exp383保存生成物を読むcontractとSHA gate、256 ft fault graph、formation AND structural cut、
  stable component、6-surface component field、最大8 componentのsoft posterior、
  base floor 0.25、prefix likelihood、exp383-compatible shrink/path、no-component exact fallback、
  target-free SHA freeze後のlate truth joinをcompact self-contained trainへ展開し、
  正規train Notebookへ採用した。専用test `14 passed`、scaffold/notebook test `11 passed`、
  Ruff、py_compile、Jupytext train/inference round-trip、strict experiment validationをPASS。
  全repository testはexp384を含む`853 passed / 6 skipped`で、未変更のexp296
  config status/run-approval期待不一致2件だけが既存FAILだった。
  exp383は後続のversion 1 Stage 0 resource FAILで閉じ、parent manifest SHAは生成されなかった。
  exp384のKaggle package/push/run、科学score、inference、submissionも未実行のまま
  `closed_by_exp383_stage0_resource_fail`とする。

- 2026-07-24、物理モデルでLB 6.5を狙う案「複数解＋GR尤度」を、候補生成と尤度評価が
  相互汚染しない2実験へ分けて設計確定した。P1
  `exp386_cycle_consistent_rgt_scenario_bank`は、6地層を個別の絶対面として補間せず、
  outer-trainの全TVTと順序付き地層区間から相対地質時間（RGT）対応グラフを作る。
  outer-valid/testの生Formation・suffix truth・GRを読まず、軌跡と既知prefixだけで
  井戸あたり8--32本のcycle-consistent pathを決定論的に固定する。Stage 1はrolling-origin
  prefix oracleでexp226比`>=0.50 ft`、4/5 folds、Stage 2はscenario oracle
  `<=5.50 ft`かつ5/5 foldsを必須とする。P2
  `exp387_prefix_gr_rgt_scenario_posterior`はexp386の全gate PASSとmanifest logical SHA固定後だけ、
  bankを変更せず、target GR level/first differenceの固定Student-t尤度とgraph-cost priorを
  exact forward-backwardで周辺化する。Stage 0はreal-vs-512-row circular controlとprefix
  heldout gain、Stage 1はpooled`<=7.20 ft`、exp226比`>=2.0 ft`、4/5 folds、
  long/hidden-like改善とnear non-regressionをAND判定する。exp386--387はexp383--385の拡張ではない
  topology-first RGT独立familyである。初回設計時は両方ともbacklog、experiment scaffold、
  steering、固定configだけを作成した。後続のユーザー指示でexp386のみ別名compact
  self-contained train候補、fail-closed inference候補、専用testまで実装した。
  後続のユーザー実行指示により、exp386は正規Notebook採用、Kaggle private CPU package/push、
  16-well Stage 0 preflight、PASS後full runまで承認した。inference、submissionと、
  exp387の実装以降は未承認・未実施を維持する。

- 2026-07-24、`exp386_cycle_consistent_rgt_scenario_bank`のKaggle private CPU
  version 1（id_no `128478384`）は16-well / 5-fold Stage 0を`2411.033 sec`で完了した。
  RGT source coverage `0.989847`、target GR / valid Formation / suffix truth read各0、
  source-valid overlap 0、projected runtime `2867.246 sec`、peak RSS `1.145931 GB`はPASS。
  しかしgraph query / scenario-bank / finite-path coverageはすべて0で、scenario count
  p05も0、cycle residual p95は`2.363303 > 0.10`だった。全16井戸でscenario bankが
  空のため、full run、Stage 1/2、edge/stretch/scenario-count/diversity救済、
  inference、submissionなしで`stage0_fail_closed`とした。必須parent bankが存在しない
  `exp387_prefix_gr_rgt_scenario_posterior`も未実装・未実行で閉じた。logsにはrouteの
  棄却段階がないため詳細原因は断定せず、topology-first RGT familyをP1からP3へ降格する。
  再訪時は同じ固定設定のedge residual成分とroute rejection funnelだけを測る
  0-prediction診断を先行し、parameter救済やfull予測へ直接進まない。

- 2026-07-24、ユーザー指示により
  `exp390_parallel_strip_surface_registration_readout`を実装し、正規train Notebook採用、
  Kaggle private CPU / internet offの16-well Stage 0 preflightまで実行した。
  query PCA axis、modulo-π pair角度、same-s donor補間、two-sided weighted Huber fit、
  prefix vertical gauge、exp226 exact fallback、Stage 0 target-free、Stage 1 rolling-origin /
  circular control、Stage 2 truth-late scoreとlogical SHAを10章のcompact self-contained
  train候補へ展開した。fail-closed inference候補と専用testも作成し、resolver回帰testを
  含む`11 passed`、Ruff、py_compile、Jupytext round-trip、strict experiment validationを
  PASSした。version 1は3件のtest inputを選ぶresolver不具合でscientific処理前に停止し、
  773件trainを件数で一意選択するfail-closed修正後のversion 2（id_no `128480051`）は
  `60.401419 sec`で`COMPLETE`。16 wells中eligible pairを持つqueryは8、全10 pairs、
  queryあたり最大2 donorsで、4 donorかつ正負両側supportを満たすnodeは0だった。
  two-sided row / well coverageとdonor p05はすべて0でStage 0 FAIL。一方、angle p95
  `1.769352°`、overlap p05`0.897013`、leakage/read 0、runtime/RSSはPASSした。
  threshold/one-sided救済を行わず、full run、Stage 1 / 2、inference、submissionなしで
  `stage0_failed_closed_sparse_two_sided_support`としてbranchを閉じた。

- 2026-07-25、ユーザー指示により
  `exp364_signed_curvature_exact_hmm`の正規train Notebookを採用し、Kaggle private CPU
  version 1（id_no `128529795`）で0-HMM Stage 0を実行した。773 wells中772 wells、
  13,631 complete blocksを`224.737080 sec`で評価し、freeze前truth / hidden-like read 0、
  SHA readback、unique key、16-well extremaを含むtechnical gateは`12 / 12 PASS`。
  top1 `0.550143`、MRR gain `0.252574`、1000+とhidden-like 2面のRMSE方向はPASSしたが、
  real-minus-circular top1は`0.003081 < 0.03`、passing foldは`3 / 5 < 4 / 5`だった。
  peak RSS `4.880433 GB`は上限内だが、固定runtime projection
  `33857.604 > 30600 sec`もFAILした。exp367と同様にnegative-control差とfold再現性が
  足りず、さらにexact state resource条件も満たさないため、Stage 1 exact HMM、
  inference、submission、parameter / emission / parallelism / blend救済なしで
  `completed_stage0_gate_failed_closed`としてbranchを閉じた。

- 2026-07-25、ユーザー指示により
  `exp366_fault_reset_duration_semimarkov_hmm`の0-HMM Stage 0を実装し、正規train
  Notebook採用後にKaggle private CPU version 2（id_no `128543224`）を完了した。
  version 1はraw well identityへCSV SHAを使ったadapter不一致で科学処理前に停止し、
  version 2では親のlogical SHA契約への修正だけを行った。`3,783,989 rows /
  773 wells`、eligible `3,389,090 rows`を`666.798832 sec`で評価し、freeze前truth /
  hidden-like read 0、SHA readback、入力identity、HMM/model/booster/control rerun 0を含む
  technical gateはPASS。発火は`40 events / 30 wells`だけで、trigger率
  `0.0000118 < 0.001`、bad-event AUC `0.500004 < 0.60`、circular差
  `0.000003 < 0.05`だった。alternative within-10 coverageは`0.90`でPASSしたが、
  GR evidence MRR gainは`-0.123356`、selected branchはbaseより`1.005307 ft`悪化し、
  passing folds `0/5`、hidden-like 2面も負方向だった。triggerとGR-only branch selectionの
  識別性がなく、`stage0_failed_close_without_semimarkov_hmm`としてStage 1、
  inference、submission、threshold/jump/duration/margin救済なしでbranchを閉じる。
  完了済みexp366をbacklogから削除し、exp289/290/231のnegative evidenceを補強する。
  同familyの救済backlogは追加しない。

- 2026-07-25、ユーザー指示により
  `exp394_soft_sticky_exp226_k16_branch_hmm`を設計確定後に実装した。
  group-safe exp226 `tvt_geop`を1-state E branch、exp355 K16 relative rateを
  GR補正前の遷移平均に使うexp209全absolute-TVT-grid exact HMMをH branchとし、
  初期50/50、base switching length 1000 MD-ft、H→E docking 6 ftのsoft-sticky regimeを
  1つのlog-space forward-backwardで周辺化する。低ランク3D地層場、有限mode bank、
  exp226 `gr_delta/tvt_pred`、Huber/Student-t、selector、後段blendは対象外。
  16-well段階はfinite / normalization / runtime / RSSだけのtechnical preflightで、
  小標本RMSEを773-well統合候補へのscientific gateにしない。full OOFは別承認後の
  1 variant / 5 folds / 773 switching-HMM runs / booster・control rerun各0とし、
  保存exp263 OOF `8.238331546`比`>=0.25 ft`とfold/stress/well-tail/branch非退化を
  promotion条件に固定した。12章/3,522行のcompact self-contained train候補、
  fixed16 technical preflight、full OOFのlate-truth/SHA/promotion orchestrationを実装し、
  optimized kernelとdense全列挙を含む専用`10 passed`、Jupytext round-trip、py_compile、
  RuffをPASSした。後続のユーザー実行指示により、正規Notebook採用、private CPU
  package/push、固定16-well technical preflightだけを承認した。実行量はtechnical
  candidate 1 / HMM well runs 16 / LightGBM config・trained fold・booster・control
  rerun・GPU各0。canonical version 1（id_no `128536142`）は16 wells /
  140,721 rowsを`3703.079064 sec`で完了した。finite/full-grid coverage、
  identity/leakage、posterior normalization、transition row sum、projected peak RSS
  `1.515934 GB`はPASSしたが、full runtime projection `112,736.889439 sec`は
  固定上限`30,600 sec`の`3.684212x`で唯一FAILした。RMSEは計算していないため
  `technical_blocker_not_scientific_negative_result`として、full OOF、inference、
  submissionなしで閉じる。exp226、exp355、exp263の既存判断は変更しない。
  Public LB 6.5は到達目標であり、現時点の実証値ではない。

- 2026-07-25、ユーザー指示によりexp394の全状態・科学契約を変えずruntimeだけを
  監査する`exp399_soft_sticky_fused_exact_runtime_audit`を実装・実行した。
  P×R×5 position tensorを廃止し、exp209型fixed-width max+sum、
  source-boundary on-demand正規化、forward/backward docking融合、2 wells ×
  Numba 2 threadsへ変更した。Kaggle private CPU version 4（id_no `128546220`）は
  fixed16 / 140,721 rows / 3,290,350,409 state-time unitsをtotal`632.688257 sec`、
  decode wall`589.600103 sec`で完了し、exp394比`6.168148x`、full projection
  `18,277.265455 sec`（`5.077 h`）、peak`2.544819 GB`で全technical gateをPASSした。
  全TVT×41 rate states、key、K16 schedule、finite/full-grid coverageを維持し、
  prediction max差`7.883838e-6 ft`、差RMSE`5.65e-7 ft`、branch probability
  max差`5.110359e-8`、diagnostic max差`2.978181e-6`だった。truth/error/hidden-like
  pre-freeze read、model、booster、control rerun、GPUは各0。version 1はkey dtype比較、
  version 2/3は事前数値閾値とKaggle CPU runtime変動を切り分け、backward融合で遅い
  runtime側にも余裕を作った。technical decisionは
  `technical_preflight_passed_full_oof_requires_separate_approval`。実装済みruntime
  backlogを削除した。full 773-well OOFはsummary SHA
  `ac800ac0...50c660`を凍結して別承認後にversion 5を実行し、773 / 773 wellsを
  `28,107.311 sec`でdecode・prediction freezeまで完了したが、exp226 foldとexp263の
  独立outer-foldを同一と誤って要求したlate-readout contractでERRORになった。
  631 / 773 wellsでfold labelが異なる一方、両ledgerは0..4 coverageとwell内一定性を
  満たす。科学式を変えず両foldを独立監査し、pre-truth frozen predictionをcheckpointする
  version 6を同一kernelで完了した。3,783,989 rows / 773 wellsのfull OOFを
  `25,118.126809 sec`で完走し、row/well identity、finite、normalization、
  transition row sum、saved exp263 parity、truth/hidden pre-freeze 0、runtimeを含む
  全technical gateはPASSした。candidate RMSE `11.395645678`はexp209
  `11.938287556`より`0.542641877 ft`改善したが、exp226 `9.427109836`より
  `1.968535842 ft`、promotion baseline exp263 `8.238331667`より
  `3.157314012 ft`悪化した。fold改善は`0/5`、nonworse wells `40.4916%`、
  1000+ `+3.488101 ft`、hidden-like spatial/typewell-purged
  `+4.385663/+4.198109 ft`、by-well p95 `+12.034886 ft`、worst
  `+38.148059 ft`で、10 scientific checksをFAILした。E/H occupancy
  `0.052990/0.947010`とswitch rate `0.348381/1000 MD-ft`は非退化だったため、
  failureはbranch collapseではなく、H優勢のsoft-sticky平均がexp263固定blendを
  置換できないことによる。decision=`full_oof_rejected_no_rescue`としてparameter救済、
  blend、selector、inference、submissionなしで科学branchを閉じる。runtime kernelは
  全状態と実用的数値同値を保つ再利用可能な基盤として残すが、この結果だけを根拠にした
  soft-sticky救済backlogは追加しない。exp226/exp263の既存判断は変更しない。

- 2026-07-25、exp209 persistent offsetの全OOF truth-late原因監査を実施した。
  `abs error >10 ft`が128行以上続く638 episodes / 807,710 rowsが全SSEの
  `91.9880%`を占める。5/10/15/20 ft × 64/128/256 rowsの12定義でもSSE占有率は
  `73.7385--98.1042%`、符号一貫性medianは全条件1.0で、定義依存ではなかった。
  episode直前128行のabs error slope medianは`0.025201 ft/row`、episode内は
  `0.002673 ft/row`で9.43倍縮み、pre128 abs change medianは`3.271 ft`だった。
  onset符号一致はepisode選択の影響を含むため因果根拠に使わないが、形成期だけrate/error
  driftが速く、その後local rateが再同期してもdatum差を戻せないramp→parallel-lock構造は
  独立な時間的証拠として維持する。
  実装式ではposition transitionが
  `K_p(p_t-p_{t-1}-(r_t*dMD-dZ))`だけに依存し、grid内部でpath全体の一定平行移動に
  厳密不変。initial position priorは開始時だけでsuffix途中のgeometry unary / datum prior /
  reset / re-anchorはなく、posterior position edge occupancyも0だった。よって一時的な
  rate mismatchでdatum差を作った後はtransitionに復元力がなく、GRだけがabsolute datumを
  戻すcontinuous translation-gauge lockをpersistent offsetの構造原因とする。exp279
  absolute unaryの平均改善とexp281 episode減少は整合するが両者のtail FAIL、exp366
  GR-only reset AUC 0.5から、単純な常時anchor/hard resetを解法とはしない。
  observed raw GRはcandidate側を強く支持する群が全SSE
  `30.7385%`でwrong-depth GR matchingを直接支持する一方、truth側を強く支持する群の方が
  `52.7585%`と大きかった。NLL閾値を`±0/1/2/5/10/20/50/100`へ変えても全条件で
  truth-support SSEがcandidate-support SSEを`9.6656--25.5224` points上回った。
  exp209の0.35 ft position gridと実効sigma floor
  0.1225 ftの5点kernelはsub-grid変位を0方向へ縮め、1行bias絶対平均
  `0.011553 ft`、最大`0.048707 ft`。episode開始前128行の累積biasとsigned offsetは
  Spearman`0.634649`、符号一致`85.1097%`、SSE加重`83.5618%`で、initial-rate
  mismatch制御後もpartial Spearman`0.551028`だった。actual motion上のkernel分散は
  current`0.004578 ft²`に対しsigma 0.2325で`0.053720 ft²`（`11.734x`）。
  minimum-variance exact-mean transportの下限は`0.005262 ft²`で、主問題は単なる
  sigma不足ではなくcoarse gridと狭いprocess noiseの一次/二次moment不整合だった。
  rate support内でbiasを補償する
  rate shiftは絶対median`0.025` / p90`0.045`で、`sig_r=0.002`と3-state rate transitionに
  対して大きい。`mom=0.998`のtruth-centered 0向きrate mean-reversionは局所絶対量が
  position biasの`0.5966%`と小さい一方、pre128 signed offsetとのSpearman`0.5744`、
  kernel bias制御後partial`0.3857`で、第二のdirection seedを支持した。
  局所分散による補償時間proxyはmedian`156.25` / p90`506.25 rows`、
  `mom=0.998`のrate半減期は約346行で、実測onset median 232行と同じ数百行スケールに
  揃った。さらにoracle補償rateがgrid edgeになる行を10%以上含む211 episodesが
  episode SSEの`40.4812%`を占め、edge sourceのoutward probability median`0.06`を
  捨てるsub-stochastic transitionの暗黙penaltyが確認された。edge率とRMSEはepisode長、
  true-rate span外、補償shift/時間を制御後もpartial Spearman`0.3052`。
  保存exp355 rate-mean介入は元persistent rowsを
  RMSE`24.7831→20.9282`、SSE`-28.6896%`まで回復したが、それ以外を
  `3.8102→6.5765`へ悪化させた。episode内の観測GR NLL差は隣接符号持続率median
  `0.8585`、lag-1相関median`0.8827`、lag 1--20の記述的IAT median`23.9247`で、
  exp343のtail raw tau約24--25と独立に一致した。ただしwrong-candidate支持run長と
  episode RMSEのSpearmanは`0.2857`で、GR自己相関は増幅因子だがseverityの単独主因では
  ない。posterior mean / marginal MAP / global Viterbiのrow-wise診断oracleはpersistent
  rowsをRMSE`24.7831→15.2329`、SSE`-62.2%`まで下げ、状態空間内により良い候補が
  残ることを示したが、posterior stdとViterbi gainのSpearmanは`0.0503`でtarget-free
  routingには使えない。rate span外10%以上の18 episodesはRMSE`37.0486`だがepisode
  SSE寄与は`6.3949%`でseverity modifierに留まる。したがって単一のGR aliasではなく、
  prefix rate prior、position-kernel量子化shrinkage、sticky rate補償、
  forward/backward basin mass、GR alias/reference mismatchの相互作用である。
  旧exp327は同じ量子化軸を仮説化したが
  未実装・未実行でnegative evidenceではない。追加の制御合成HMMではgeology/GR aliasを
  除き、exp209の41 rate states、0.35 ft position grid、5点kernel、forward-backwardを
  維持して、position sigma・rate境界再正規化・momentumの2×2×2にexact-mean
  transport 4条件を加えた12 variants × 8 scenarios = 96 casesで分離した。
  centered 2 ft emission、true rate 0.05のfiltered RMSEは
  current`0.450709`、position sigma 0.2325単独`0.080201`、boundary単独`0.418143`、
  momentum=1単独`0.381468`、position+momentum`0.007005`、全補正`0.001720 ft`。
  exact-mean transportも`0.129908 ft`へ改善し、filtered position stdは
  current / sigma / exact-mean=`0.471374 / 0.671527 / 0.518113 ft`だった。
  neutralの512行目誤差はsigma / exact-mean=`-10.0565 / -9.9631 ft`とほぼ一致し、
  両者が同じposition mean biasを除く一方、sigmaの追加改善は拡散強化を含むと分離した。
  neutral emissionではcurrentの512行目誤差が`-20.5025 ft`まで蓄積し、遷移だけでも
  符号付きdriftを作ることを確認した。一方、centered emission下のsmoothed RMSEは
  filtered`0.450709`から`0.057524 ft`へ改善したため、backward smoothing演算自体が
  単独root causeではなく、actual大offsetには構造化GR/missing/path multiplicityが必要。
  position端mass最大`2.2825e-10`、符号対称誤差`1.0658e-14 ft`、
  stochastic-neutral message差最大`1.57e-8 ft`で実装不変条件を確認した。
  また全3,783,989 suffix transitionsのraw dMDはmin/maxとも厳密に`1.0 ft`で、
  clamp発火・不規則stepは0行、固定dMD=1の合成条件と実データが一致した。
  773実grid上の初期position prior bias最大は`6.94e-17 ft`、初期rate prior bias最大
  `4.00e-7`、rate内部遷移の意図meanとの差最大`3.47e-18`で、31,693 source rows中
  probability floor / cap発火は各0。初期prior加重のrate境界mass lossもmedian
  `2.74e-13` / max`4.01e-6`だった。よってinitial anchor/prior離散化とrate内部遷移式は
  root causeから除外し、rate境界はposteriorが補償stateへ移動した後のseverity modifierとする。
  未正規化float32 log-messageも同じsparse transitionの512-step制御HMMでstressし、
  全state共通`-3.5 / row`によりmessage絶対値を約1,824まで増やしても、正規化float64比の
  filtered / smoothed mean最大差は`1.52e-4 / 1.48e-4 ft`、正負非対称最大
  `1.81e-4 ft`だった。保存posterior meanの実値域でのfloat32 round誤差上限も
  `0.0004883 ft`。実offsetより4--5桁小さく、message scaleと保存精度をroot causeから除外する。
  さらに全397 experiment configs / Python sourceをcensusし、HMM transition関係61 configsの
  数値contractは`sig_p=0.02`が59/59、step`0.35`が54/54、momentum`0.998`が61/61、
  Python `sig_p` literal 67件も全て`0.02`だった。position変更exp327とmomentum変更exp326は
  ともにclosed-without-runで、完了actual介入は0件。既存結果の見落としではなく、
  actual position/momentum interventionは未実行と確定した。
  さらに各wellの実prefix initial rateと実suffix dMD/dZだけを使い、GR/suffix truthなしで
  transition-only priorを全773 wellsへ伝播した。current priorの診断RMSEは`39.783886 ft`で、
  prior errorとactual HMM errorのwell内相関medianは`0.542651`だった。一方、
  exact-mean positionは`95.236351 ft`へ悪化し、そのepisode効果とactual mean errorは
  Spearman`-0.465604`、符号一致`32.4451%`。有限position band逸脱を避けたanchor-safe
  239 episodesでもSpearman`-0.134265`、符号一致`43.0962%`だった。rate境界補正は
  効果絶対median`0.113334 ft`、Spearman`-0.030364`、momentum=1は同`3.005631 ft`、
  Spearman`-0.234944`で、単独補正のactual offset方向との対応はなかった。
  したがってtruth-centered局所監査と正しいrate開始の制御合成HMMは、position
  shrinkage/mean-reversionがoffsetを作れる条件付き機構を示すが、実prefixでは現行
  shrinkageがstale/misspecified rate外挿を抑えるregularizerにもなる。position kernelを
  actual offsetの無条件root causeとはせず、actual filtered rate posterior、
  predictive/alpha/beta basin mass、GR更新との相互作用を未確定中心とする。
  既存3 episode ledgerを完全一致keyでjoint readoutすると、current prefix prior errorと
  actual offsetはSpearman`0.612816`、符号一致`75.3918%`で、一致481 episodesが
  episode SSEの`85.3208%`を占めた。prior絶対error median`36.7507 ft`に対しactual HMMは
  `13.9152 ft`で、`80.8777%`のepisodesではHMMがpriorより真値へ近づけるが、同方向残差を
  取り切れない。prior符号一致はcandidate-strong / near-tie / truth-strongで
  `75.56 / 71.67 / 77.70%`、SSE加重`86.45 / 89.30 / 84.02%`とGR classをまたいで残った。
  排他的joint taxonomyでは、prior-aligned + truth-GRが216 episodes /
  episode SSE`48.1910%`で最大、同群SSEの`86.2858%`はViterbiで部分以上に回復可能。
  同群は全row emissionでもSSE`99.87%`、imputed-onlyでも`92.16%`がtruth-strongで、
  prefix static affine後も`90.04%`がtruth-strong。補間GR aliasとstatic calibrationでは
  説明できない。
  prior-aligned + candidate-GRは136 / `28.8866%`、prior-aligned + near-tieは129 /
  `8.2432%`。一方、prior-opposed + truth-GRは62 / `9.1626%`で、prefix priorとobserved
  raw GRのどちらも実offsetを説明せず、全rowはSSE`100%`、imputed-onlyも`88.48%`が
  truth-strong、affine observedも`91.23%`がtruth-strongだった。imputation/static affineを
  除き、beta・joint path multiplicityの
  最優先診断群とする。
  保存decoderのepisode平均符号も再集計し、marginal MAPはposterior meanと
  `99.5298%`同符号、5 ft以内は`2.1944%` / SSE`0.4138%`。global Viterbiは
  `60.6583%`のepisodes / SSE`81.9481%`でabs mean errorを縮めるが、`77.4295%` /
  SSE`74.8571%`では同符号のままで、5 ft以内は`21.4734%` / SSE`20.0658%`に留まった。
  Viterbi headroomを完全な別mode脱出とは読まず、stable mode ID / hard decoder置換だけでは
  continuousなrate/history offsetを解消できないと判断する。
  さらにexp270 exact joint top-5を638 persistent episodes内で再監査した。5 unique
  full-well pathsが利用可能なのは439 episodes / SSE`77.5628%`だが、episode平均TVT spanは
  median`0` / p90`0.001029 ft`、pairwise path RMSE最大もmedian`0` / p90`0.019545 ft`。
  `70.6897%`のepisodesでは利用可能top-Kがepisode内で完全一致し、top-1との平均datum差が
  1 ftを超えるalternativeは1 / 638、5 ft超は0だった。persistent pooled RMSEはtop-1
  `19.842201`、episode best-of-5`19.841305`、row-wise top-K oracle`19.841294 ft`で、
  追加4 ranksのheadroomは約`0.0009 ft`。truthをtop-K min/maxが挟むrowも`0.01164%`だけ。
  rank-2 score gap median`0.001038`とほぼ同点でも、別macro datumではなく数行1-gridの
  micro-path degeneracyがrankを消費する。よってexp270 top-K rankをmode IDとして保持する
  familyは回復先を持たず閉じるが、lower-ranked macro basinのsum-product総質量までは
  反証しない。`path_log_posterior`ではtop-1 joint path probabilityのwell中央値が
  約`10^-466.20`、top-5合計でも約`10^-465.51`。長さ正規化top-1 surprisalは
  persistent / nonpersistent wells=`0.21851 / 0.22176 nats/row`で、persistent 450 wells内の
  RMSE / persistent-row率とのrhoも`0.0047 / -0.0246`だった。single path IDはbasin massを
  運ばず、genericなpath diffusenessもseverityを説明しない。未観測中心はtruth/wrong
  macro basinに属するmicro-path群の相対総質量であり、top-K拡張ではなくStage Aのbasin
  集約massを必要条件とする。
  exp209 smoothed posterior geometryも同じpre-onset ringで監査した。全体のposterior std
  episode平均は256--512行前`3.1616`、直前16行`3.9575`、episode内`4.8353 ft`。
  eligible 619 episodesの`71.2439%` / SSE`78.1202%`でnear stdが増え、mode massは
  `0.2667→0.2516`、mean--MAP距離は`1.6242→2.0747 ft`へ動いた。最大の
  prior-aligned + truth-GR群ではstd`4.0654→5.3874→7.0471 ft`、near broadening
  `79.2453%` / SSE`81.7982%`。一方opposed-prior + candidate-GR群は
  `1.1914→1.1579→1.4609 ft`で集中したまま。よってconfident wrong-GR basinと、
  truth GRにhistory massが抗して広がるconflicted basinを別regimeにする。std変化とRMSE /
  transition crescendoのrhoは`0.0919 / 0.0414`なのでbroadeningをseverity rootにはせず、
  smoothed値だけではalpha起源かbeta起源かも断定しない。
  同じringでtruth TVTと凍結posterior meanの一階差分をrateへ分解した。全体のtrue
  `|rate|`は256--512行前`0.03647`から直前16行`0.04318`へ増えた一方、decoded
  `|rate|`は`0.03517→0.03095`へ減り、absolute rate errorは
  `0.01581→0.03365`、pooled比`2.2848x`へ増えた。eligible 619 episodesの
  `78.9984%` / SSE`85.5844%`でerrorが増え、transition NLL crescendoとのrho
  `0.6812`、pre128 slope絶対値とは`0.6463`。true accelerationは
  `0.00850→0.00813`、増加episodes`47.8191%` / SSE`45.8826%`なので、一般的な
  hard curvature spikeを反証し、rate lagを直接の運動学的形成機構とする。最大の
  prior-aligned + truth-GR群ではtrue `|rate|``0.03760→0.04723`に対しdecoded
  `|rate|``0.03314→0.02434`で、zero-directedなunder-responseを強く支持する。
  opposed-prior + candidate-GR群は逆にtrue `|rate|``0.03258→0.02933`中にdecoded
  `|rate|``0.04330→0.06129`となり、wrong-GR駆動の別経路である。rate errorはTVT
  errorの微分と最大差0であり、最終offsetとの符号一致`98.4326%`は定義・onset selectionの
  影響を含む。smoothed mean差分ではhidden rate massやalpha/betaを分離できないため、
  この時点ではposition shrinkage、momentum、sticky rate transitionの寄与率を
  Stage Aへ残し、後続exp408でactual messageを分離した。
  保存decoder別にも同じrate-lag監査を行った。合法なglobal Viterbiのabsolute rate errorは
  far`0.02151`からnear`0.03196`へ`1.6477x`、eligible episodesの`65.2666%` /
  SSE`56.2309%`で増え、transition crescendoとのrhoは`0.8364`。最大prior-aligned +
  truth-GR群でも`0.02208→0.03292`、`1.6848x`、rho`0.8382`なので、rate lagを
  sum-product posterior meanの平均化だけに帰す説を反証する。一方nearではViterbiがmeanより
  episodes`49.8433%` / SSE`66.3421%`で改善するが、episode全体では
  `32.2884%` / SSE`37.7751%`しか改善せず、長区間の単純decoder置換にはならない。
  transition/history basinを共通root、sum-product mass / mean readoutを重い形成期の
  経路依存増幅器とする。row-wise marginal MAPはfar/near rate error
  `0.04530→0.06853`で既知のhard違反・grid jumpに支配され、stable mode carrierから除外する。
  行単位でもtrue rate非zeroのrowsをzero-directed under-response / opposite /
  same-direction overshoot / tieへ排他的分解した。episode内778,966 moving rowsを分類し、
  28,744 zero-rate rowsを方向比率から除外、class count parityを全区間で確認した。
  最大prior-aligned + truth-GR群のnearはposterior meanでzero-directedが
  rows`73.0826%` / rate-error mass`64.6574%`、合法Viterbiでも
  `64.1990% / 59.3099%`で、0向き追従不足が群平均の見かけでないことを確立した。
  mean zero-directed row率は5 ringsで`59.74→61.70→66.06→71.10→73.08%`と
  onsetへ単調増加した。
  一方opposed-prior + candidate-GR群のnearはsame-direction overshootがmean
  `93.4524% / 96.5197%`、Viterbi`77.2321% / 82.0325%`で、wrong-GRに整合する
  high-rate pathへの別経路を確立し、mean overshoot率も
  `54.61→77.74→86.28→87.56→93.45%`とonsetへ増加した。prior+wrong-GR同方向群ではzero-directed
  `68.9882%`なのでcandidate-GR全体をovershootとはせず、prefix basinとGR方向のjointで
  分類する。dominant群でもopposite-direction error mass`27.2114%`が残り、そのhidden
  state起源はこの時点ではStage Aへ残し、後続exp408でforward hysteresisを主因、
  backward reversalとmultiplicityを増幅器へ分類した。
  同じringのraw-observed row countからmissingness timingも監査した。missing率は全体で
  far`31.6220%`からnear`30.3977%`へ微減し、near-minus-far変化とmean rate-error増分 /
  transition crescendo / RMSEのrhoは`-0.0275 / -0.0217 / 0.0461`。最大prior-aligned +
  truth-GR群も`33.6709→32.3785%`、rate-error増分とのrho`0.0447`で、missing gapへの
  突入を一般onset triggerから除外する。opposed-prior + candidate-GR群は
  `15.8887→15.1989%`、near全missing 0で、直接wrong-GR経路は観測GRが十分ある状態でも
  発火する。一方missingが10 points以上増えるepisodesは`18.9015%` / SSE`24.6968%`、
  25 points以上`5.3312%` / SSE`7.0792%`なので、subsetのanchor低下・severity modifierには
  残す。near全missingは1 / 638 episodes / SSE`2.0581%`だけ。
  exp270の保存path diagnosticsにはrow-level latent rate pathはないが、global Viterbiの
  rate-state switch率とtop-5 rate-path SHAが残っていた。switch count復元誤差最大
  `9.27e-13`で、全773 wellsのmedian/p90は`0 / 2`、`75.0323%`がsuffix全体zero-switch。
  persistent 450 wellsも`71.5556%`がzero-switchで、persistent episodesの
  `67.3981%` / SSE`73.2917%`がそのwell上にある。top-5 rate hashが1種類のwell上にも
  episodes`68.4953%` / SSE`77.2609%`があり、最大prior-aligned + truth-GR群では
  群SSE`79.6846%`。switch率とRMSE / mean rate-error増分 / transition crescendoのrhoは
  `-0.0298 / -0.0410 / -0.0068`で、latent rate-mode switchは必要条件でもseverity原因でも
  ない。stable max-product rate IDは大半で既に保持されており、必要なのはabsolute position
  basinのsum-product massとbasin内conditional rate momentである。rate path本体とposteriorは
  persist禁止だったため後続exp408のStage Aを要し、現在は完了済みである。
  exp270 aggregateの14 artifactsも最終censusし、row-level candidateはposition候補・
  posterior geometryの17列だけ、decoder manifestは`persist_rate_paths=false` /
  `persist_full_posterior=false`、path diagnosticsのrate情報はwell-level switch率とSHAだけ
  と確認した。exp209側にもalpha / predictive / beta / latent-rate保存物はなく、既存artifact
  再集計でmessage寄与をさらに確定する余地はない。
  position効果`current-exact`はcurrent priorとSpearman`-0.922645`。prior-aligned
  85.3208% SSE群ではactualとのrho`-0.747570`、符号一致`10.8108%`、
  SSE加重`4.9190%`で保護側、prior-opposed 14.6792%群ではrho`+0.724288`、
  符号一致`98.7261%`、SSE加重`99.8309%`で増幅側だった。よってtruth-centered
  kernel biasの高相関を平均因果効果とは読まず、position shrinkageをrate basin依存で
  符号反転するmodifierへ再分類する。actual passでは同一のfiltered source-rate massを
  固定したcurrent/exact-mean 1-step期待変位差を保存し、rate feedbackからposition
  momentだけを分離する。
  current position shrinkageを残した`boundary-normalized + momentum=1` priorは
  episode abs mean error medianを`36.7507→29.1207 ft`へ下げ、`65.9875%` episodes /
  SSE`72.4382%`でcurrentより真値へ近づいたが、差分とactual offsetはrho`0.150641`、
  符号一致`58.3072%`と弱い。boundary / momentum単独では方向対応がなく、position
  exact-meanまで含む全補正はmedian`45.4590 ft`へ再悪化した。rate境界とmean-reversionは
  joint modifier候補としてedge/rate posteriorを先に観測し、盲目的なvariant実行はしない。
  このprefix-prior attributionをtail-30 estimator bugとは読まない。actual HMMでwindowを
  32/64/128/256へ変えたexp268は423/773 wellsが同一rate、spread p90`0.02`、
  best direct gain`0.042706 ft`、whole-well oracle gain`0.097314 ft`だけだった。
  static window選択より、suffix内のrate非定常性とGRによるrate posterior再捕捉を原因中心とする。
  candidateからtruthへの41点GR NLL landscapeも全638 episodesで再構成した。
  observed端点NLLは既存ledgerと最大`4.55e-13`で一致。pointwise truth morphでtruth終点が
  良い374 episodes / episode SSE`60.9894%`に限定しても、途中barrier medianは
  `61.8245 NLL`、20 NLL超がSSE加重`95.3217%`だった。最大のprior-aligned +
  truth-GR群216 episodesではbarrier median`74.2071`、20超が群内SSE`95.3054%`。
  candidate shapeを保って平均offsetだけ除くconstant datum shiftは、終点GRが改善するのが
  212 / 638 episodes / episode SSE`18.1542%`に限られたが、そのsubsetでもbarrier median
  `41.4616 NLL`、20超がSSE加重`83.4798%`だった。したがって正しいdatumが終点で有利でも
  GRは単調な復元力にならず、absolute datumとrate/local phaseが結合した非凸basinを作る。
  barrierとactual RMSEのrhoはpointwise / constant=`0.4570 / 0.4757`、Viterbi gainとは
  `0.2003 / 0.1943`で、metastability要因ではあるがseverityの単独主因ではない。
  これはtruth-late固定sliceでminimum-action HMM pathではないため、actual escape確率とは
  読み替えない。
  truth path自体のstate/transition grammarも全773 wells / 3,783,989 rowsで監査した。
  local illegal率はpersistent / nonpersistent=`0.01424% / 0.01431%`で同等、onset前128行に
  hard breakがあるのは638 episodesの`2.3511%` / episode SSE`3.4525%`、隣接rate
  continuityによる追加breakは0だった。よってtruthがhard support外へ出ることは全体原因から
  除外する。一方、前行のlocal position-conditioned rate分布をexact 3-state kernelで
  1 step伝播したtruth two-step NLLはpersistent / nonpersistent=`0.24785 / 0.23458`、
  global Viterbi persistent=`0.10851 / row`だった。onset前の非重複ringでは256--512行前
  `0.24656`から直前16行`0.37585`へ上昇し、全ringを持つ619 episodesの
  `67.2052%`でnear > far、near-far差とpre128 error slope絶対値のrhoは`0.4204`。
  episode RMSEとのrhoは`0.1408`、Viterbi gainとは`-0.0866`に留まるため、soft grammar
  mismatchをrate lagの形成トリガーとして支持するがseverityの単独説明にはしない。
  row-wise marginal MAPはpersistent episodesの`53.9185%` / SSE`87.5560%`で少なくとも
  1 hard違反を持つ一方global Viterbiは0であり、marginal mode列をstable mode IDとして
  保持する案も不適切。これはGRなし・truth-late・二段局所診断なので、actual filtered rate
  massでの因果確定は後続exp408のmessage保存Stage Aで完了した。
  同じ非重複ringのraw観測GRだけをexp209 emissionで再採点すると、
  `truth NLL - posterior-mean NLL`は256--512行前`-0.00979`から直前16行
  `-0.02460 / observed row`へ動き、全体平均では発症へ近づくほどtruth側を支持した。
  transition near-far増加とGR near-far変化のrhoは`-0.0552`、GR変化とpre128 error
  slope絶対値も`-0.0723`で連動しない。near/far双方に観測がある618 episodes中、
  transition負荷が上がる415 episodesでは、`54.9398%` / 条件付きSSE`66.0339%`が
  直前16行でもtruth側GRだった。最大のprior-aligned + truth-GR群は
  `60.9589% / 64.9977%`で、rate/history lagがwrong-GRより先に発火する経路を強く支持する。
  一方opposed-prior + candidate-GR群では直前もcandidate側が
  `70.3704% / 73.8243%`で、wrong-GR initiatorの別経路も維持する。wrong-GRを全体原因には
  せず、直接initiatorとなる群と、transitionで形成後に相関evidence・非凸barrierでlockを
  増幅する群へ分ける。このtimingも保存smoothed pathとerror-defined onsetのtruth-late
  診断であり、この時点ではalpha内の更新順序が未確定だったが、後続exp408で
  predictive / filtered / backward寄与を分離した。
  また最終episode平均offsetをsuffix先頭から直接持つ仮想pathのinitial position prior追加
  NLLはmedian`172.12` / p90`695.00`。一方pre128 error slope median
  `0.02520 ft/row`はrate grid約`5.04 cells`で、5 ft形成時間約198行が実onset median
  232行と揃う。別datumは先頭から選ばれるより、rate mismatchで徐々に形成され、その後
  translation gaugeと非凸GR landscapeでlockする像を強める。
  この未観測message診断は後続exp408で完了した。450 wells / `2,264,135 rows`を
  1 current variantだけで再生し、exclusive SSEはforward / backward / multiplicity /
  support / mixed=`59.40 / 23.04 / 9.04 / 6.39 / 2.12%`、raw-GR / imputation alias 0%
  だった。予測約3時間に対する実測は`4.425 h`、peak`3.588 GB`。position width /
  exact-mean / momentumの盲目的介入へは進まず、rate prior / sticky transition追従遅れを
  主因とする。以後の介入は単一のtransition / reset仮説、新番号、事前gate、
  ユーザー承認を必要とする。

- 2026-07-25、ユーザー指示により
  `exp395_left_right_mode_consensus_confidence_readout`をdesign-onlyで確定した。
  exp386はscenario bank coverage 0でFAIL_CLOSE、exp387も親不成立で閉鎖済みのため
  再利用せず、exp209 exact-HMMとexp391 stable transition-overlap mode lineageを
  共通mode carrierにする。exp391のtruth-free 1,234 eventsをprimary checkpointとし、
  左右512-row GR windowをcheckpointから各64 rows離して重複をなくす。
  `sum_m min(P_L(m),P_R(m))`だけをprimary confidence、`1-overlap`をbad10 risk scoreに
  固定し、right GRの2048-row circular shiftを単一negative controlとする。
  予測値、mode選択、補正、blend、selector、inference、submissionは作らない。
  exp391 Stage A1全PASSを実装前go/no-goとし、PASS後も固定16-well Stage 0、
  full 773-well confidence readoutはそれぞれ別承認を必要とする。現時点では
  backlog、steering、template scaffoldだけで、Notebook/helper/test/package/runは0。

- 2026-07-25、ユーザー指示により
  `exp396_fold_safe_exp111_score_27_addonly_on_exp287`を設計し、後続指示でStage A候補を実装、
  正規train notebookへ採用してKaggle private CPUの0-booster preflightまで実行した。
  exp111保存済みfold0 scorerを全trainへ適用した旧27列は再利用せず、downstream outer 5 foldsの
  各outer-train内で4 GroupKFold × binary/L1の2目的を学習するstrict nested設計とした。
  Stage Aは40 CPU boostersで、outer-trainをinner OOF、outer-validをouter-train由来4 model平均に
  固定する。全technical / scorer-quality / resource gateをPASSした場合だけ、exp287の
  421特徴へ27列を追加するStage Bを別承認候補にする。Stage Bは1 variant × 3 configs ×
  5 folds = 15 GPU boosters、control再学習0、最終448特徴。targetを特徴生成前に分離する
  10章compact self-contained train候補、fail-closed inference候補、固定48/10/27 schema、
  stable sample、model固有median、artifact SHAと全AND gateを実装した。専用10 tests、
  py_compile、Ruff、Jupytext round-tripはPASS。Kaggle version 1（id_no `128540844`）は
  3,783,989 rows / 773 wells / 20 nested fold roles、well overlap 0、coverage 1.0、
  fixed 48/10/27 schemaを含む16/16 checksをPASSした。runtime `277.133756 sec`、
  peak RSS `5.168308 GB`、booster/prediction/submission/control再学習は各0。
  後続指示で固定40 CPU fitを実行し、version 2（同id_no `128540844`）で40/40 models、
  40/40 medians、10 partitions、technical 22/22をPASSした。runtime `3662.974058 sec`、
  peak RSS `8.762432 GB`。expected-error MAE、within10 logloss、within10 Brierはpooledかつ
  5/5 foldsでpriorより改善し、Stage A全gate PASS。後続指示でStage B実装と
  1 variant × 3 configs × 5 folds = 15 T4 GPU boosters、control再学習0を承認した。
  保存済みformation 10 partitionsとStage A score core 10 partitionsをSHA固定して再利用し、
  421 + 27 = 448列の固定add-only学習、exp287/exp264比較promotion gateを実装した。
  別Stage B kernel version 1（id_no `128570498`）で15/15 boostersを完走し、
  OOF `8.134294735`、exp287比`-0.002413486 ft`だったが、必要値`<= -0.02 ft`に届かなかった。
  foldは2/5のみnonworse、scope最大`+0.026155871 ft`、by-well p95`+0.342926545 ft`、
  corrected exp264比worst`+7.802733095 ft`。+1/+3/+5 ft悪化well数`68/16/5`だけが上限を
  PASSし、固定promotion gateは1/6 PASSで総合FAIL。scorer-qualityのStage A PASSは
  downstream TVTの安定価値へ転移しなかった。score-27 add-only familyはnegative resultとして
  backlogから削除し、exp287をtrain-side parent anchorに維持する。subset/grid、same-OOF rescue、
  gate緩和、再学習、inference、submissionへ進まない。次の証拠候補は救済ではなく、
  保存済みStage A/Stage B生成物だけでscorer-qualityからdownstream TVTへの転移失敗を分解する
  0-booster readoutに限定し、既存P3候補を追い越さない低・P4へ置く。

- 2026-07-25、ユーザー指示により
  `exp397_prefix_gr_agreement_adaptive_sigma_exact_hmm`をdesign-onlyで確定した。
  exp209のknown-prefix zero-fill residual `sigma_gr` とGaussian absolute-TVT exact HMMを親にし、
  raw finite known-prefix horizontal GRとtypewell GRのPearson相関だけをwell-level reliability
  gateにする。pair 64以上かつ両std `>1e-6`で `rho_gr < 0.50` のwellだけ、exp209の
  `[10,60]` clip後scaleを `1.3` 倍し、その他と判定不能wellは `1.0` のno-opとする。
  Stage 0はfull prefix / last-512 stability、coverage、係数非退化を確認するtruth-free 0-HMM監査。
  全gate PASS・別承認時だけStage 1を1 variant / 5 reporting folds / 最大773 HMM runs、
  model・booster・PF・Beam・parent control再実行各0で評価する。`1.0` wellはsaved exp209
  predictionを再利用する。exp209比`>=0.05 ft`、4/5 folds、changed-group p95、worst、
  stress 6面、fixed LikPF 50:50を全AND gateに固定した。exp307/346のscale縮小、
  exp343のACF tempering、exp389のHuber救済ではない独立仮説である。threshold、multiplier、
  support、window、bias、emission、blendの事後救済は禁止。その後のユーザー指示でStage 0を
  compact self-contained candidateとして実装し、full/tail agreement、係数SHA freeze、
  truth-read 0、7条件AND gate、fail-closed実行を専用11 testsで固定した。Jupytext conversion、
  py_compile、Ruff、strict experiment validationはPASS。さらに正規notebookへ採用し、
  Kaggle private CPU version 1（id_no `128540665`）を`39.35975061899995 sec`、
  diagnostic 1 / reporting folds 5 / HMM・model config・trained fold・PF・Beam・booster・
  parent control再実行各0で完了した。773 wellsすべてでfull/tailを評価でき、primary coverage
  `1.0`、fallback `0.0`、fold minimum coverage `1.0`、truth-read before freeze 0、
  input/coefficient SHAはPASS。一方、`rho_gr < 0.50`は`8/773 = 0.0103493`で固定下限`0.10`を
  FAILし、full/tail multiplier agreement `0.666235 < 0.80`、Spearman
  `0.167466 < 0.70`もFAILした。coverage不足や実装異常ではなく、binary reliability surfaceが
  ほぼno-opかつtail不安定という科学的negativeである。decisionは
  `stage_0_failed_close_without_rescue`。threshold/multiplier/support/window/correlation種の
  調整、Stage 1、inference、submission、version 2、同family rescue backlogなしでbranchを閉じる。

- 2026-07-25、ユーザー指示により、exp397を再開・再分類せず、
  `exp398_all_well_1p3_sigma_gr_exact_hmm`を独立した固定global interventionとして採番・実装した。
  exp209のknown-prefix zero-fill population `sigma_gr`を`[10,60]`でclipした後、全773 wellsで
  `1.3`を正確に1回掛け、再clipなしの`[13,78]`とする。Gaussian `z²` clip 600、
  absolute-TVT grid、41 rates、transition、prior、missing-GR、Type Well補間、posterior meanは
  exp209のまま。saved exp209 HMMをcontrolとして再実行せず、候補だけを1 variant /
  773 HMM well-runs / 5 reporting folds、model config・trained fold・booster・PF・Beam・
  control rerun各0で評価した。overall`>=0.05 ft`、4/5 folds、raw-observed`>=0.05 ft`、
  raw-missing/high-missing/1000+/hidden-like 2面/by-well p95/worst/fixed LikPF 50:50を
  AND gateに固定。exp389 compactと同じ10章のself-contained train/inference候補、
  truth-late freeze、input/prediction SHA、専用9 testsを実装し、pytest、Jupytext、
  pycompile、Ruff、strict validationはPASS。その後の実行指示により正規notebookへ採用し、
  private CPU version 1（id_no `128542706`）を`19324.104 sec`で完走した。候補RMSE
  `12.710664`はsaved exp209 `11.938287`から`0.772377 ft`悪化し、改善は1/5 folds、
  330/773 wells。raw-observed `-0.592611`、raw-missing `-1.150295`、high-missing
  `-1.559979`、1000+ `-0.862967`、hidden-like 2面`-2.053945 / -2.109669 ft`、
  fixed LikPF 50:50も`-0.383411 ft`で全required scopeが悪化した。by-well p95
  `+7.038260 ft`、worst `+46.046495 ft`もFAIL。実行済み倍率監査のfalseは、CSV再読込値を
  `atol=0`で比較した最大`2.13e-14`のround-trip差による偽陰性で、全773記録の倍率`1.3`、
  effective sigma `14.551610--78.0`は成立。ローカル監査へ`atol=1e-12`を追加したが、
  科学結果は独立して明確に悪化しているため再実行しない。decision
  `all_well_sigma_x1p3_failed_close_without_rescue`として、multiplier/clip/emission/
  transition/grid/blend救済、inference、submission、version 2なしでbranchを閉じる。
  exp397/398の連続negativeを踏まえ、exact-HMM GR sigma multiplier familyの追加backlogは作らず、
  既存候補の優先順位は変更しない。PF固有の逐次重み更新・resamplingへ同じ倍率を適用する仮説は、
  HMM救済ではなく別algorithmの独立実験として扱う。

- 2026-07-25、ユーザー指示により
  `exp400_all_well_1p3_sigma_gr_likelihood_pf`をdesign-onlyで確定した。Kaggle discussion
  728712で共有された公開Notebook後半`lik_pf`の`gs * 1.3`を、同型kernelを持つexp072
  deterministic v2へ移す。全773 wellsで
  `gs_candidate=1.3*clip(prefix zero-filled GR residual population std,10,60)`、
  再clipなしの`[13,78]`とし、500 particles、128 stable per-well seeds、scale
  3/5/8/12、dynamics、likelihood、resampling、補間は固定する。primaryは保存exp072
  `likpf_mean`比で、scale 4出力はbest選択しないsecondary diagnostic。実行量は1 variant /
  773 PF well-runs / 98,944 seed-well trajectories / 49,472,000 particle starts /
  5 reporting folds / model・booster・HMM・Beam・parent PF control再実行各0。
  overall`>=0.05 ft`、4/5 folds、raw observed/missing、high-missing、1000+、
  hidden-like 2面、by-well p95/worst、saved exp209-HMMとのfixed 50:50を全AND gateにした。
  現行リンク先Notebookでは最終selector用の別`run_particle_filter`はx1.0のままなので、
  full public pipeline scoreの再現とは主張しない。backlog、steering、template scaffoldだけを作り、
  Notebook/helper/test/package/run/inference/submissionは別承認まで0とする。

- 2026-07-25、ユーザーの「exp400を実装してください」によりexp400の
  implementation-onlyを完了した。正規Notebookは上書きせず、11章のcompact
  self-contained train候補、submission非生成のfail-closed inference候補、
  exp072 x1.0 synthetic fixture exact parityを含む専用test 9件を追加した。
  固定raw SHA `14faee3...`のexp072 cacheには`likpf_mean_d`だけがあり、
  x1.0 scale 3/5/8/12列は保存されていないため、x1.3 scale別出力は
  candidate-only nonselective diagnosticとして扱う。primary saved
  `likpf_mean` gateは変更せず、parent PF再実行0を維持する。正規Notebook採用、
  package、push、PF実行、inference、submissionは別承認待ち。

- 2026-07-26、exp400のKaggle private CPU version 1（id_no `128585102`）を
  `10496.299889 sec`で完走した。technical gateはPASSし、773/773 wells、
  98,944 seed-well trajectories、49,472,000 particle startsをfallbackなしで
  実行、input / multiplier / truth-late / execution count / control parity /
  artifact SHAはすべて成立した。一方、primary `likpf_mean_x1p3` RMSE
  `12.221811`はsaved exp072 `11.594894`から`0.626917 ft`悪化し、改善は
  1/5 folds・305/773 wellsだけだった。raw observed/missing、高missing、
  1000+、hidden-like 2面は`-0.453077 / -0.998656 / -0.884439 /
  -0.708353 / -0.706604 / -0.738688 ft`、fixed exp209-HMM 50:50も
  `-0.390275 ft`で全required scopeがFAIL。by-well p95 / worst regressionも
  `+5.059698 / +32.160524 ft`だった。secondary scale 3/5/8/12は
  `11.271336 / 11.174615 / 11.243685 / 11.342899`だが、x1.0 scale controlが
  なくpost-hoc best選択は禁止なので救済に使わない。探索的にはhigh-missing /
  high-base-scale wellsほど悪化したが、同じOOFのadaptive multiplierへ流用しない。
  decision `all_well_likelihood_pf_gs_x1p3_failed_close_without_rescue`として、
  version 2、inference、submissionなしでbranchを閉じる。exp398と合わせて
  global sigma multiplier familyを追加救済せず、既存候補の優先順位は変更しない。

- 2026-07-26、ユーザー指示により
  `exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264`を
  design-onlyで確定した。exp368のmarginalized-PF branchはknown-prefix
  NLL gain`0.037356% < 1%`、weak mass`0.009689 < 0.02`のFAILを維持する。
  一方、saved suffix bad10 AUC`0.636675`、circular差`+0.058264`、
  5/5 folds、hidden-like`0.641795 / 0.636115`を、GR likelihoodを弱める
  thresholdではなくexp264 selector用の連続risk候補として切り出した。
  Stage 0はoverlapするexp368 blockの`weak_posterior_mean`をrow算術平均し、
  truth前にreal/circular feature SHAをfreezeする。exp264 corrected Stage C v6の
  45,407,868 candidate-long rowsを使い、`likpf_mean` bad10 rowで既存
  `pred_abs_error`が指名したother candidateがwithin10へ回復するかを、
  primary 11候補domainとsecondary 7候補domainを分離してreadoutする。
  pooled AUC、circular差、4/5 folds、hidden-like 2面、既存selector
  margin条件付きAUC、Q4-Q1 realized advantageを全AND gateにした。
  Stage 0は1 diagnostic / 5 reporting folds / model・LightGBM・trained fold・
  booster・PF・prediction各0。全gate PASS・別承認時のStage 1も
  1 variant / 1 LightGBM config / 2 objectives / outer 5 × inner 4 =
  40 CPU selector boosters、parent control再学習0に固定し、downstream TVT、
  inference、submissionはscope外とした。backlog、steering、template scaffoldと
  設計文書だけを作り、実装・Notebook採用・package・runは行っていない。

- 2026-07-26、ユーザーの「exp401を実装してください」によりexp401の
  Stage 0 implementation-onlyを完了した。正規Notebookは上書きせず、
  11章 / 2,054行のcompact self-contained train候補、submission非生成の
  fail-closed inference候補、専用contract test 9件を追加した。
  exp368 overlap blockのreal/circular row平均、exp226 fold-only projection、
  exp264 corrected Stage C v6の45,407,868行Parquet row-group scan、
  primary 11 / secondary 7候補domain、other-4-fold margin decile / weak
  quartile、truth前のfeature/schema/surface/scientific-contract SHA freeze、
  late truth後のAUC / margin-conditional AUC / Q4-Q1と全AND gateを実装した。
  candidate TVTは一時float32 memmapで保持しreadout後に削除する。
  Jupytext、pycompile、Ruff、strict validation、`9 passed`を確認した。
  Stage 0実行量は1 diagnostic / 5 reporting folds / model・LightGBM・
  trained fold・booster・PF・prediction各0のまま。正規Notebook採用、
  package、push、run、Stage 1の40 CPU boosters、inference、submissionは
  別承認待ちで、未着手backlogから実装済みtrain待ちへ移した。

- 2026-07-26、ユーザーの実行承認によりexp401の正規train Notebookを採用し、
  Kaggle private CPU version 4（id_no `128626512`）を`129.300203 sec`、
  1 diagnostic / 5 reporting folds / 45,407,868 candidate-long rows /
  model・LightGBM・trained fold・booster・PF・prediction各0で完走した。
  version 1はsafeな`pred_abs_error`のtruth guard誤判定、version 2はexp264
  generation foldとexp226 reporting foldの誤同一視、version 3は成果物保存後の
  `numpy.bool_`表示だけで停止し、いずれも設計・gateを変えないtechnical fixと
  regression test後に進めた。version 4は入力SHA、3,783,989 rows / 773 wells /
  15,174 blocks、12 candidates、5 folds、truth-before-freeze 0、feature /
  selector SHAを含むtechnical 15/15 checksをPASS。primary 859,755-row cohortの
  recovery10 AUCはreal/circular `0.520214 / 0.523467`、差`-0.003253`、
  margin-conditional `0.458846`、hidden-like spatial/typewell-purged
  `0.527468 / 0.513626`で固定gateをFAILした。Q4-Q1 realized advantageは
  `+3.879372 ft`、fold AUC countはPASSしたが、scientificは4/12 PASSのため
  `stage_0_failed_close_without_rescue`。output SHAを取得ファイルで照合し、
  threshold/反転/bucket/domain/subset/gate救済、Stage 1の40 CPU boosters、
  downstream TVT、inference、submissionなしでbranchを閉じ、train待ちから削除した。

- 2026-07-26、ユーザー指示により
  `exp402_fold_safe_grwr_5_addonly_on_exp287`をdesign-onlyで確定した。
  exp264 availability auditで無効だったGRWR 6列を一括復旧せず、
  formation依存の5列だけをexp287のfold-safe formation roleから再計算する。
  固定8候補はtarget-freeな`pf_ancc / beam_mean / likpf_mean / sc_ens / hyb`と、
  matching outer-roleの`tvt_dense / tvt_densew / tvt_dense50`。
  float32 `ddof=0`の標準偏差、range、既存clean-273内のDWT/FFT/NCC成分との
  固定3 interactionをexp287 421列へadd-onlyし、最終426列とする。
  親はexp287、clean tail controlはcorrected exp264、保存済みcontrol再学習0。
  0-booster preflightでouter-train self-exclusion、outer-valid outer-train-only、
  raw current-test all-train reference、target formation read 0、schema/content SHA、
  旧GRWR値/score不使用をAND確認する。PASSと別承認後の学習量は
  1 variant / 3 LightGBM configs / 5 folds = 15 GPU boosters。
  promotionはexp287比pooled`<=-0.02 ft`、4/5 folds、全scope`<=+0.02 ft`、
  by-well p95`<=0`、corrected exp264比worst`<=+0.25 ft`、+1/+3/+5 ft
  悪化well数`<=135/39/14`の全AND。exp396 entropy依存の6列目、
  score-27、sample weight、error-segment weight、hard gate、direct correction、
  同一OOF救済は対象外。backlog、steering、template scaffoldと設計記録だけを作り、
  実装・Notebook採用・package・runは行っていない。

- 2026-07-26、ユーザーの「exp402を実装してください」によりexp402の
  Stage 0 implementation-onlyを完了した。正規Notebookは上書きせず、
  11章のcompact self-contained train候補、fail-closed inference候補、
  専用test 8件を追加した。旧exp218 generator全体を呼ばず、SHA固定した
  source/configから必要なDWT/FFT/NCC 3成分だけを同式で再生成し、
  synthetic parityを確認した。exp287 matching outer-role 10 partitionの
  train self-exclusion / valid outer-train-only境界、固定8候補のfloat32
  `ddof=0` GRWR-5、schema/content SHA、raw current-test all-train
  formation reference / target formation read 0を実装した。Stage 0の
  current-test量はPF ANCC 3、PF Z 3、Beam 21 paths、likelihood-PF
  3 well-runs / 384 seed-well trajectories / 192,000 particle starts、
  model・booster・final prediction・submission各0。Jupytext、pycompile、
  Ruff、strict validation、`8 passed`を確認した。正規Notebook採用、
  package、push、run、Stage 1の15 GPU boosters、inference、submissionは
  別承認待ちで、未着手backlogから実装済みpreflight待ちへ移した。

- 2026-07-26、ユーザーの「実行してください」によりexp402の正規train
  Notebookを採用し、private CPU Stage 0 version 1（id_no `128627922`）を
  pushした。最終statusは`KernelWorkerStatus.CANCEL_ACKNOWLEDGED`、保持ログは
  15.354秒の本処理直前まででtracebackなし、Kaggle公開outputは0件。
  run/check時刻からruntime上限が最有力だが、APIは手動cancelと区別しない。
  preflightはPASS/FAILではなく未完了として扱い、同一versionの再pushはしない。
  retry前にrole partition生成とcurrent-test PF/Beam replayの分割またはcache化、
  実行量とSHA境界を再設計し、ユーザー承認を得る。Stage 1の15 GPU boosters、
  inference、submissionは引き続き未承認。

- 2026-07-26、ユーザーの「設計変更と再実行を進めてください」により、
  exp402 Stage 0を0A train source+10 roles、0B current-test 3 wells replay、
  0C immutable SHA aggregateの3 private CPU runへ分割した。候補、式、fold、
  dtype、seed、logical-content SHAは不変。専用testはaggregate synthetic
  file SHA guardを含む10件PASS。A/B/C packageはconfig SHA
  `98dd377e...6176`、implementation source SHA`665f41ad...fb20`で同時凍結した。
  0A version 1（id_no `128687498`）はRUNNING。0B初回pushはexp410の4 shardと
  0AでKaggle CPU 5枠が埋まり受理されなかったため、別実験をcancelせず空き待ち。
  0CはA/B PASS後だけpushする。Stage 1、inference、submissionは未承認。

- 2026-07-28、exp402のtrain source、current-test、outer-fold 0–4の
  upstream 7 runとaggregate version 2を完了した。aggregate version 1は
  fold 4の旧slugと同名sentinel fallbackの曖昧性だけで失敗し、config/source
  SHAを変えないwrapper-only path aliasで修正した。version 2は
  `18/18` technical checks、10 outer-role partitions、current-test
  `14,151 rows / 3 wells`をPASSし、historical GRWR load、target formation read、
  model、booster、prediction、submissionはいずれも0。partition / preflight /
  reproducibility manifestのSHAを取得してStage 0 PASSを確定した。
  次の1 variant / 3 configs / 5 folds = 15 GPU boosters、control再学習0は
  明示承認待ちで、CV、inference、submissionは未実行。

- 2026-07-28、ユーザーの実装・実行承認によりexp402 Stage 1を実装し、
  1 variant / 3 configs / 5 folds = 15 T4 boosters、control再学習0で
  canonical kernelへpushした。version 2はStage 0 aggregate inputの実mount
  pathを発見できず、10.6秒・0 boosterでtechnical failure。科学仕様を変えず、
  required manifest名と固定file SHAでaggregate/fold/exp287 rootを選ぶresolver、
  前処理前の物理T4 guard、回帰testを追加したversion 3を同一kernelで再実行し、
  `RUNNING`を確認した。専用testは13件PASS。inferenceとsubmissionは
  promotion PASS後の別承認を維持し、継続監視は行わない。

- 2026-07-28、exp402 Stage 1 version 3は物理Tesla T4 ×2とSHA-qualified
  mountを確認後、clean-273再構築に必要な`exp145-train`が未添付だったため
  227.3秒・0 boosterでtechnical failureとなった。科学仕様、特徴、fold、
  LightGBM config、promotion gateは変えず、exp145を11番目のinputへ追加し、
  固定11 inputと必要3ファイルを前処理前に検証するfail-fast guardを追加した。
  Jupytext、pycompile、Ruff、strict validation、専用test 13件、package SHA監査を
  PASSし、同じcanonical kernelへversion 4をT4指定でpush、`RUNNING`を確認した。
  承認済み学習量は15 T4 boosters、control再学習0のままで、継続監視は行わない。

- 2026-07-26、物理モデルを優先した最終TVT改善結論を検証する
  `exp403_exp333_exp355_tail_constrained_physics_shrink`をdesign-onlyで確定した。
  exp263の固定`0.50 exp226 + 0.25 LikPF + 0.25 exp209`から、K16 50%成分を
  exp333、exact-HMM 25%成分をexp355へ置換し、
  `candidate=exp263+lambda_fold*(full_replacement-exp263)`だけを検証する。
  保存済みOOFのread-only根拠ではfull置換が`8.238331745→8.159425494`
  （`-0.078906251 ft`）だった一方、改善3/5 folds、by-well p95
  `+1.983209 ft`、worst `86454a6f +13.412007 ft`だった。λは固定
  `0,1/64,1/32,1/16,1/8,1/4,1/2,3/4,1`から、各reporting foldの
  outer-trainだけでpooled/scope/p95/worstを満たす最大positive値を選び、
  なければ0へfail closedする。exp226 reporting foldとexp263 generation foldは
  独立ledgerとして保持し、既知631/773 wellsのlabel不一致をjoin errorにしない。
  実行量は1 policy / 9 calibration lambdas / 5 reporting folds /
  model・LightGBM・booster・PF・HMM・Beam・parent rerun各0。
  routeはexp333 ML補正と物理候補が本質的に混ざるため`ensemble`。
  steering、template scaffold、configと設計記録だけを作り、実装、Notebook採用、
  package、run、inference、submissionは別承認まで行わない。

- 2026-07-26、ユーザーの実装指示によりexp403の凍結済み設計だけを実装した。
  exp263 generation fold単位のstreaming load、exp333 / exp355のglobal key join、
  input SHA、exp226 / exp263 formula parity、source/formula/content SHA freeze、
  freeze後のraw suffix truth / hidden-like role attachment、固定9 λのouter-train
  最大eligible選択とzero fallback、fold/scope/by-well/persistent-offset/
  512-row recoveryの全AND gateを、約2,000行・10章の別名compact self-contained
  train候補へ実装した。fail-closed inference候補とsynthetic contract test 10件も
  作成し、py_compile、Ruff F821/F811、pytestをPASSした。正規Notebook採用、
  Kaggle package / run、inference、submissionは別承認待ちで、未着手backlogから
  実装済み・未実行へ移した。

- 2026-07-26、exp403のKaggle CPU trainをversion 4で完走し、
  technical gateは全PASS、scientific promotionはFAILした。full両置換の
  referenceは`8.238331667→8.159425494`だったが、各outer-trainで最小positive
  λ`1/64`のgainが`0.005785--0.007919 ft`と固定下限`0.01 ft`未満で、
  by-well delta p95も`+0.023577--+0.026743 ft`と非悪化制約を全foldで破った。
  positive eligible λは`0/5 folds`、λは全fold 0へfallbackし、cross-fit CVは
  controlと同じ`8.238331667`、gain 0となった。source freeze後だけ
  `3,783,989` truth rowsを読み、pre-freeze truth access 0、runtime
  `172.418 sec`、peak RSS`1.921 GB`、prediction/gate SHAを記録した。
  同一OOFのλ/weight/gate/router救済、inference、submissionは行わずterminal
  closeし、未着手backlogから削除した。原因追跡は独立した0-predictionの
  component別tail attributionだけを低優先度候補とする。

- 2026-07-26、exp404 Kaggle private CPU version 4（id_no `128628818`）が
  COMPLETEした。version 1は2 variants / 1,546 PF well-runs /
  197,888 seed-well trajectories / 98,944,000 particle startsを完走し、
  prediction freeze後のlate-readout config欠落で停止した。version 2 / 3の
  gzip suffix推論とpandas dtype表記差もtechnical recoveryで解消し、version 4は
  同一prediction bytes、logical SHA
  `5f4b6e715081b598b0a34607ad0c81339d0ecd5882ea3a45dd79f33123959a00`、
  scientific contractを保持してlate readoutを完了した。technical gateと
  exp072/exp400 parityは全PASS。一方scale5 x1.0 / x1.3 pooled RMSEは
  `10.914522 / 11.174615`で、x1.3が`0.260093 ft`悪化した。nonworse foldは
  `1/5`、raw-GR observed / missing、high-missing、1000+も悪化し、
  by-well p95 `+4.826467 ft`、worst `+37.333851 ft`だった。hidden-like 2面の
  小幅改善だけでは事前固定した全AND gateを満たさないためscientific FAIL。
  global scale5 x1.3は同じOOFで救済せずterminal closeし、inferenceとsubmissionは
  行わない。

- 2026-07-26、保存済み12物理pathのinterval semi-Markov融合を検証する
  `exp405_geometry_reinjected_interval_semimarkov_fusion`と、その科学FAIL時だけ
  開く`exp406_loop_closed_multiwell_rgt_fixed16_stage0`をdesign-onlyで確定した。
  exp405はexp293 candidate値・順序・H256/H512 block・SHAを固定し、
  candidate周囲`±55 ft / 5 ft`のType Well GR morphology、
  minimum H512 duration、exact posterior meanを1本だけ評価する。
  新segmentの`exp226_k16` prior floorは0.10で、docking、trigger、
  GR likelihood、現在modeと独立。exp297 evidence、exp399 docking transition、
  exp370 trigger resetは再利用しない。合格は`<=6.90 ft`、exp263比5/5 folds、
  1000+ / hidden-like 2面、well-tail、real対2 controls、geometry massを全ANDとし、
  PASS後だけ同じexp405でcurrent-test実装資格を得る。技術的に有効なscientific
  FAIL時だけexp406を解禁し、technical ERRORでは解禁しない。
  exp406はexp386 fixed16 selectorだけを再利用し、Formation graphではなく
  H256/H128 horizontal-GR pairwise edgeをTVT ft単位でloop-closeする。
  Stage 0はgraph/cycle/circular control/prefix512/resourceだけで、
  unknown suffix predictionは0。両expともtemplate scaffoldのみで、
  実装、package、run、current-test、inference、submissionは行っていない。

- 2026-07-26、ユーザーの「exp405を実装してください」によりexp405の
  implementation-onlyを完了した。正規Notebookを上書きせず、10章・約2,750行の
  compact self-contained train候補とdedicated synthetic test 11件を追加した。
  exp293 matrix / manifest / block assignmentのSHA parity、pre-truthの
  `MD/GR/TVT_input`限定readとraw truth file SHA遅延、H256 block-local
  raw/roll21/roll101 morphology、Laplace 23-shift周辺化、candidate-common
  neutral mixture、SHA256固定circular / full-block permutation controlを実装した。
  exact explicit-duration forward-backwardはminimum 2 blocks、final short
  right-censor、uniform duration、log9 switch penalty、docking-independent
  geometry floorを保持する。block-center weight interpolation、convex hull /
  continuity guard、score/posterior/3 predictions freeze後のtruth/hidden-role
  readout、constrained oracleと全AND gateまで実装した。fixed16はfold別SHA256
  rank round-robinの`4/3/3/3/3` wellsへ固定した。py_compile、Ruff
  F821/F811、Jupytext round-trip、strict validation、`11 passed`を確認した。
  fixed16 / fullの実行flag、正規Notebook採用、package、current-test、
  inference、submissionは閉じたままで、未実装から実装済み・未実行へ移した。

- 2026-07-26、後続の「実行してください」によりexp405の正規train Notebookを
  採用し、fixed16 Kaggle CPU preflightだけを実行した。59文字の初回
  slug/titleはKaggle SaveKernel 400で未作成だったため、意味を保った50文字の
  `exp405-geometry-reinjected-semimarkov-fusion-train`へid/titleを同時にそろえ、
  version 1（id_no `128631270`）を完走した。`81,485 rows / 16 wells /
  12 candidates / 3 controls`でtechnical gate 13項目を全PASSし、実測
  `24.578047 sec / 1.317642 GB`、full投影
  `1,187.426900 sec / 1.822834 GB`だった。summary SHA256は
  `78774852751fcb534f528938f03006c97aecfe0c516359144f8f11cd2826a9c6`。

- 2026-07-26、exp405のtechnically-valid scientific FAILで解禁された
  `exp406_loop_closed_multiwell_rgt_fixed16_stage0`のimplementation-onlyを完了した。
  exp386同型round-robin fixed16、exp226 fold、exp065 native-overlap Type-Well、
  H256/H128、±55/5 ft、12 donors、top4 edge、raw/roll21/roll101 NCC、
  SHA256 circular control、fundamental-cycle Huber IRLS 10回、target-free SHA freeze、
  prefix512/resource全AND gateを3,171行・10章のcompact self-contained train候補へ
  実装した。保存exp226 OOFがprefix行を持たないため、ユーザー承認済みの推奨案として、
  graph freeze後にouter-trainからoriginal K16 field/Kappaを5 folds以内で再構築し、
  fixed16 pseudo-cutの`tvt_geop`相当だけをcontrol化する。target ANCC、
  GR correction、U-projection、official OOF再生成は0。fail-closed inference候補、
  dedicated test 13件、py_compile、Ruff F821/F811、Jupytext round-trip、
  strict validationをPASSした。正規Notebook採用、Kaggle package/push/run、
  full OOF、current-test、inference、submissionは未承認のまま。
  これはresource / leakage / numerical integrityだけのpreflightであり、
  scientific gateは未評価。fixed16実行flagを閉じ、full saved-OOFは別承認待ち、
  current-test / inference / submissionは引き続き未承認とした。

- 2026-07-26、ユーザーの「full oofを実行してください」により、同じexp405
  canonical Kaggle CPU kernel version 2（id_no `128631270`）で
  `3,783,989 rows / 773 wells / 12 candidates / 3 controls`を完走した。
  runtime `1,434.099 sec`、peak RSS `2.220737 GB`、technical 17/17と
  constrained oracle 2/2はPASSした。oracleは`3.606822 ft`だったが、
  posterior meanはanchor `8.238332 ft`に対して`8.451060 ft`
  （`+0.212728 ft`）で、全5 foldsと1000+ / hidden-like 2面を悪化させた。
  real gainはcircular `0.000346 ft`、block permutation `0.000230 ft`だけで、
  morphology evidenceはnegative controlとほぼ区別できなかった。geometry mass
  3 gateはPASSしたため、失敗原因はgeometry occupancy不足ではなく、
  candidate bankのoracle headroomを識別できないevidence側と判断する。
  decision SHA
  `e159cfb712a6ed81e78f4524febbf0d995375124a473a5056aad3c1347b648f0`
  でscientific FAILを確定し、same-OOF rescueなしでexp405を閉じた。
  current-test / inference / submissionは禁止し、独立familyのexp406 Stage 0を
  実装可能状態へ解禁した。

- 2026-07-26、`exp406_loop_closed_multiwell_rgt_fixed16_stage0`のKaggle private
  CPU version 1（id_no `128637170`）を固定16 wells、5 graph contexts、
  1 diagnostic、model・booster・PF・HMM・Beam各0で完了した。target-free
  elapsedは`1,356.649 sec`、peak RSSは`0.544994 GB`。technicalは12/15 PASSで、
  graph query coverage `0.451157 < 0.90`、finite loop-closed row coverage
  `0.755026 < 0.95`、773-well runtime投影
  `65,543.109 sec > 30,600 sec`の3項目がFAILした。connected coverage 1.0、
  9,272 fundamental cycles、cycle residual p95 `70.0 -> 7.1e-15 ft`、
  real-circular NCC差`+0.874148`と5/5 folds real優位はPASSしたため、
  GR signal欠如やloop solver不安定ではなく、固定pairwise構築のsupport不足と
  計算量を主因と判断する。target側rejectionはnonpositive local TVT progress
  43.97%、NCC閾値未満19.79%、finite pair不足16.24%、retained 0.63%。
  target-free gateで停止したためprefix truth joinとexp226 K16 replayは0回で、
  prefix科学性能は未評価。leakage/read 0、unknown suffix prediction 0、
  生成物manifest file SHA 8/8一致を確認した。固定decision
  `close_exp406_without_parameter_rescue`に従い、fixed16再選択、
  donor/window/shift/edge/NCC/Huber/gate救済、prefix-only rerun、full OOF、
  current-test、inference、submissionなしでbranchを閉じる。exp406項目は
  backlogから削除し、exp386 route棄却の独立原因分解だけをP4候補として残す。

- 2026-07-26、ユーザーの「exp407を実装してください」によりexp407の
  Stage B implementation-onlyを完了した。正規Notebookを上書きせず、親trainと
  同じ465行・8章のcompact self-contained train候補を別名で作成した。
  fit labelsだけを入力にするfold別inverse-RMSE weight、最終`[0.5, 1.5]`
  range fail-closed、両objective共通sample weight、unweighted validation、
  sampling / truth-read / feature content / model / OOF SHA監査を実装した。
  保存済みcorrected exp264 Stage B v5とのfold / near / 1000+ / hidden-like /
  worst-well全AND gateと専用test 9件も追加した。Stage B 10 CPU boosters、
  control再学習0は維持した。その後ユーザーの実行承認を受け、正規Notebookへ採用して
  private CPU version 1をpushした。ユーザーの完了連絡後に同versionを監査し、
  `1,531.430秒`で`COMPLETE`を確認した。technical gateは全PASSした一方、
  expected-error MAE `3.798670`（親比`+0.002869`）、within10 logloss
  `0.360461`（`+0.000489`）、Brier `0.112648`（`+0.000197`）、
  hard-primary RMSE `8.668141`（`+0.081137`）でscientific全AND gateはFAILした。
  1000+ `+0.091228`、hidden-like spatial `+0.103759`、
  typewell-purged `+0.079052`、worst well `+16.226863`もFAILした。
  gate SHA `2ae8cb3e...99962`、decision
  `fail_close_exp407_without_rescue`で閉鎖し、weight/clip/exponent/candidateの
  救済探索は行わない。Stage C/D、inference、submissionも閉じ、
  corrected exp264 Stage B v5をselector anchorとして維持する。

- 2026-07-26、ユーザーの「そのバックログを実装してください」により、
  `exp409_saved_selector_candidate_switch_tail_attribution_on_exp407`の
  implementationを完了した。保存済みcorrected exp264 Stage B v5と
  exp407 Stage B v1のcandidate-score OOF SHA、12候補順、11候補hard domain、
  fold、distance bucket、hidden-like assignment、worst well `52f1e77a`を固定した。
  1,329行・8章・17セルのcompact self-contained候補で、Phase 1はtruth列を拒否して
  両surfaceのselectionとtransition/scopeをfreezeし、SHA確定後のPhase 2だけで
  `actual_abs_error`を読み、加法的な`exp407 SSE - parent SSE`をtransition /
  fold / distance / hidden-like / well別に帰属する。magnitude thresholdは置かず、
  同じpositive excess-SSE rank-1 transitionが1000+とhidden-like 2面で各4/5 folds、
  固定worst wellでもrank-1かだけを判定する。synthetic test 9件、py_compile、
  Ruff、Jupytext round-trip、strict validationをPASSした。exp407は結果にかかわらず
  scientific FAILのまま再開しない。その後ユーザーの実行指示を受け、親OOFをprivate
  Dataset化し、canonical private CPU version 1（id_no `128678587`）を完了した。
  3,783,989 rowsのうち1,289,588 rows（34.0801%）でcandidateが変化し、
  overall RMSEはparent `8.587004`、exp407 `8.668141`（`+0.081137 ft`）。
  1000+ `+0.091232`、hidden-like spatial `+0.103759`、
  typewell-purged `+0.079052 ft`だった。固定worst wellでは
  `exp226_k16__selfgr_hmm_a070 -> likpf_mean__exact_hmm`がpositive excess-SSEの
  85.99%を占めたが、同遷移は1000+で1/5、hidden-like 2面で0/5 foldsしか
  rank-1にならず、全tail scope各4/5を満たすtransitionは0件だった。
  decision `diffuse_or_nonreproducible_candidate_switch_cause`でgate FAIL、
  exp407のscientific FAILを維持したまま原因分解branchを閉じる。exp409項目は
  backlogから削除し、selectorのsame-OOF救済案は追加しない。次の優先候補は
  独立familyのP2第一案
  `exp411_predictive_filtered_rate_innovation_destick`。正規Notebook採用とStage 0
  Kaggle CPU Version 5を完了した。technical gateは13 / 13 PASSしたが、
  future-rate方向一致`0.225397`、passing folds`0 / 5`、
  control active-row fraction`0.136119`、persistent-control active-well差`0.0`で
  mechanism gateは2 / 6 PASS、`stage0_fail_closed`。Stage 1、inference、
  submissionへ進まずbranchを閉じる。P3第二案
  `exp412_beta_filter_rate_disagreement_two_pass_reset`は先行条件を満たし、
  2026-07-28の実装・実行指示でKaggle CPU Version 3を完了した。方向一致
  `0.776347`、4 / 5 folds、forward / control安全性はPASSしたが、backward cause
  SSE reductionは`-0.069575`、full runtime投影は`51,753.199秒`でFAIL。
  `stage0_fail_closed`としてStage 1、inference、submissionなしでbranchを閉じる。

- 2026-07-27、`exp410_likpf_particle_resampling_basin_audit`を原因診断として完了した。
  exp072 likelihood-PFのPF固有persistent offsetを496 wells / 839 episodes /
  819,288 rowsへ固定し、500 particles ×128 stable seedsのexact replayをKaggle CPU
  4 shardで実行した。496 / 496 wellsで保存`likpf_mean`との最大差`0.0 ft`、
  strict coverage / SHA / duplicate guardをPASS。排他的SSE比はfinite particle
  support不足`36.4701%`、across-seed算術平均`36.2441%`、within-seed particle平均
  `10.8561%`、transition`10.7177%`、GR alias`3.6664%`、resampling extinction
  `0.7577%`。真値がmajority particle support外の行は`64.2853%` / SSE
  `83.0651%`だがhard clamp外は0、resampling直後の平均移動は`0.000245 ft`、
  majority-seed extinctionは0だった。exp408 HMMとはPF SSEの`78.9383%`を占める
  区間が重なり、誤差方向`90.2655%`一致だが内部mechanism一致は`8.4071%`だけ。
  HMMはforward transition / priorとbackward smoothing、PFはfinite supportと
  particle / seed basin平均という別経路で同じ曖昧区間にoffsetを維持する。
  full結果前に固定した12 sentinel wells ×12 paired variantsもKaggle CPU 4 shard、
  144 well-runsで完了し、baseline parity 0と全guardをPASSした。roughening 10倍は
  episode SSE`0.752997倍`、process noise 3倍は`0.891691倍`だが、改善は
  `10/16・8/12`、`11/16・8/12`で符号検定非有意かつtarget-late。GRほぼ無効は
  `8.835072倍`、process noise 0は`6.265855倍`、resampling無効は`3.480912倍`、
  clamp 2倍は全16 episodes同値、particle mode / seed medianも改善しなかった。
  よって直接resampling extinction、hard clamp、GR全体、単純平均readoutだけを
  root causeとする説を棄却し、有限粒子supportとbasin平均を主因、resampling時の
  多様性 / genealogyを不均一な増幅・回復レバーとする。prediction candidate、
  inference、submissionは作らず、roughening / process noiseを検証する場合だけ
  保存exp072 control再学習0・単一固定variant・全OOFを別承認で行う。

- 2026-07-27、exp407の悪化をさらに分解し、candidate×fold平均score shiftだけなら
  親`8.587004`から`8.580477`へ改善する一方、平均shiftを除いたrow-local成分は
  `8.673599`へ悪化、final weightとrow-local score差stdのSpearmanは
  `-0.593387`と確認した。したがって候補別RMSE自体ではなく、inverse-RMSEを
  共有selectorのtask weightへ変換したことによる局所gradient / splitと
  candidate rankingの崩れをexp407の主因とする。この原因を避けるexp415では、
  candidate RMSEをfold-safe additive priorとして候補方向だけに使い、
  親TVTからの補正を各行`±0.25 ft`へ制限した。Kaggle private CPU version 1
  （id_no `128717911`）でtechnical 15/15、scientific 6/6をPASSし、
  RMSE `8.587004 -> 8.563474`（`-0.023530 ft`）、5/5 folds、4/4距離bucket、
  hidden-like 2/2を改善した。785 scopesのrisk inequalityも全PASSし、
  worst-well悪化は`+0.171379 ft`で固定上限`+0.25 ft`内だった。
  decision `rmse_prior_bounded_nudge_method_confirmed_on_saved_oof`として、
  「候補別RMSEをsample weightにせず方向priorとして使い、補正量を数学的に
  制限する」方法を保存OOF診断上で確立した。exp415は完了し、同じOOFでの
  係数 / blend / cap救済、route anchor更新、current-test、inference、
  submissionへは進めない。

- 2026-07-27、ユーザー判断によりHMM、likelihood-PF、exp226を段階的な別候補ではなく、
  1本の物理PFへ統合する`exp420_exp226_hmm_guided_defensive_mixture_pf`を
  PF route P1として設計確定した。HMMはexp209互換untreated forward filterの
  predictive-to-filtered rate innovationをexp411固定CUSUM
  （drift`0.01`、threshold`1.0 rate cell`、activation`32`、refractory`128`）
  でschedule化するだけで、posterior mean / backward message / absolute pathは使わない。
  exp226はfold-safe GR補正前`TVT+Z` geometry局所rateだけを使う。inactive proposalは
  元transition`0.5`＋geometry `1x/4x/16x`各`1/6`、activeは元transition`0.5`＋
  geometry 3成分各`1/12`＋`mu0+direction*0.005`中心HMM 3成分各`1/12`とする。
  `p0/q` clipなし・構成上上限2、x1.0 raw-GR、500 particles×128 seeds、
  temperature-5 full-suffix evidence weightingを固定する。最終predictionはPF 1 variant
  だけで、blend / selector / MLは0、routeは`pf_beam`。Stage 0はexp411 fixed32と
  exp410 sentinel12の重複なし44 wellsで、HMM方向 / lead / control発火とPF support /
  episode SSEをAND判定し、selection-biased pooled RMSEはpromotionへ使わない。
  PASS・別承認後だけ同じexp420の773-well / 4-shard fullへ進む。fullは保存exp404比
  `>=0.10 ft`・4/5 folds、exp410 support外率`>=5 points`減、exp410 / exp408 episode
  SSE`>=10% / >=5%`減、scope / tailをmechanism gate、exp226比`>=0.03 ft`・3/5を
  standalone、exp263固定物理blend比`>=0.03 ft`・3/5をphysical-anchor gateとする。
  active scientific variant 1、Stage 0 HMM/PF`44/44`、full`773/773` well-runs、
  LightGBM config / trained fold / booster / model / GPU各0、親control再実行0。
  その後の実装承認により、compact self-contained train候補、untreated HMM
  schedule、inactive / active proposal、fixed44 / full orchestration、truth-late
  readout、fail-close gate、専用12 testsを実装した。all-guidance-zero exp404と
  HMM-weight-zero exp419のsynthetic RNG / prediction parityはbitwise一致し、
  Jupytext、py_compile、Ruff F821、strict validate-expをPASSした。正規Notebook、
  package、Kaggle run、inference、submissionは行っていない。exp419 / exp411 standaloneは
  proposal / schedule実装参照として保持し、統合版の分解比較が必要になるまで保留する。

- 2026-07-28、`exp411_predictive_filtered_rate_innovation_destick` Stage 0を
  Kaggle private CPU Version 5（id_no `128773391`）で完了した。fixed32の
  32 / 32 HMMを`1,133.132777秒`、peak RSS `1.020561 GB`で完走し、technical gateは
  13 / 13 PASS。schedule / prediction各156,088 rows、well metrics 32 rows、
  trigger truth-late 633 rows、episode lead 25 rowsの実ファイルSHAはログと一致した。
  一方、future-rate方向一致`0.225397 < 0.60`、passing folds`0 / 5 < 4 / 5`、
  control active-row fraction`0.136119 > 0.10`、persistent-control active-well差
  `0.0 < 0.20`でmechanism gateは2 / 6 PASS、`stage0_fail_closed`。
  triggerは早いが非特異的で方向証拠がなく、Stage 1、inference、submissionなしで
  branchを閉じる。exp411は完了済みのためbacklogから削除する。
  exp412もKaggle private CPU Version 3（id_no `128917257`）でbaseline 32 +
  treatment 32 HMMを`2,142.435153秒`、peak RSS `0.991058 GB`で完走した。
  technical gateは12 / 13、mechanism gateは5 / 6 PASS。active
  `5,902 rows / 21 wells`、beta方向一致`0.776347`、4 / 5 folds、
  forward SSE regression`+0.013257`、control delta`+0.005836 ft`はPASSしたが、
  主目的のbackward cause SSE reductionは`-0.069575`でFAILし、full runtime投影も
  `51,753.199 > 30,600秒`だった。改善は`fae0c593`と`57f05c51`へ集中し、
  `a9c9b150`と`c9e980e8`などの大幅悪化を相殺できない。future betaのrate方向は
  多くのactive rowで正しくても、固定10% de-stickはwrong position basinを
  安定修復しない。same-OOFでtrigger / transferを救済せず、Stage 1、inference、
  submissionなしでbranchを閉じ、完了済みのためbacklogから削除する。exp420は同一scheduleと
  同一direction / control gateを使うため、現行契約のStage 0 prerequisiteが既にFAIL。
  高コストPFをrunせず、compact実装を参照として保持する。

- 2026-07-29、`exp419_exp226_guided_defensive_mixture_pf`をKaggle private CPUの
  preflight version 1、full 4 shards version 1、merge version 1
  （merge id_no `128974840`）で完了した。3,783,989 rows / 773 wells、
  proposal allowlist、freeze前truth read 0、`p0/q`最大`1.999999999999981`、
  geometry-weight-zero差`0.0 ft`、preflight/full fixed-well byte parityを満たし、
  technical gateはPASS。candidate RMSEはexp404 `10.914522`から`10.680074`へ
  `0.234448 ft`改善し4/5 folds、raw-GR observed `0.165544 ft`改善、
  persistent episode SSEも`14.8213%`削減した。一方、主目的のmajority-seed
  predictive support外率は`64.2061% -> 97.4973%`で`33.2912 points`悪化し、
  hidden-like spatial `-0.115823 ft`、by-well p95 `+5.766213 ft`、
  worst well `+20.570238 ft`、exp226比`-1.252965 ft / 1 of 5 folds`でFAILした。
  importance correctionでtarget posteriorを保っても、500粒子の半数をglobal geometry
  proposalへ割く有限粒子配置は安全でない。mechanism / standalone adoption gateとも
  FAIL、`proposal_rejected_close_without_same_oof_rescue`でterminal closeし、
  完了済みのためbacklogから削除する。exp420のgeometry proposal側とexp432の
  reinjection familyへnegative evidenceを反映し、inference / submissionへ進めない。

- exp418 Stage 0 version 1は0 boosterで3,783,989 rowsを完走し、signed-rate oracleは
  exp226 RMSE `9.427110`を`0.646951`へ改善、5/5 foldsで科学閾値を満たした。一方、
  matrix / sequential integration差`6.2954e-12 ft`が事前固定`1e-12 ft`を超え、
  technical 8/9 PASSで`FAIL_CLOSE_BRANCH`となった。同一OOF後の閾値緩和をせず
  exp418を閉じ、Stage 1 / inference / submissionは行わない。
- exp421はexp418をtechnical FAILのまま維持し、truth-free `1e-10 ft`
  numerical contractをPASSした。version 1 / 3は0 boosterのtechnical guardで停止し、
  version 2 / 4の0-booster診断でtrain/current-test SHA scopeと
  in-memory/persisted nested serialization boundaryを特定・固定した。Kaggle private
  CPU version 5（id_no `128915070`）は5 boostersを約1,176秒で完走。CVはexp226
  `9.427110`から`9.405572`へ`0.021537 ft`改善したが、改善は2/5 folds、
  1000+ delta `+0.003414 ft`、by-well p95 `+0.513310 ft`、worst
  `+10.467233 ft`で、pooled / exp228 / exp333 gateもFAILした。8 PASS / 7 FAILの
  `FAIL_CLOSE_BRANCH`としてinference / submissionなしで閉じる。
- `exp424_exp209_momentum1_exact_hmm_ablation`はKaggle private CPU
  Version 1（id_no `128924158`）でexp209
  `mom=0.998` baseline 32 + `mom=1.0` treatment 32 HMMを
  `2,077.533832秒`、peak RSS `1.030926 GB`で完走し、technical gateは
  13 / 13 PASSした。rate under-response SSE shareは`46.601854%`から
  `36.751859%`へ`9.849995 points`改善し、matched control pooled delta
  `-0.054769 ft`、by-well p95 `+0.157066 ft`もPASSした。一方、主目的の
  persistent episode SSE削減は`0.475550% < 5%`、改善wellは
  `8 / 16 < 10 / 16`、改善foldは`3 / 5 < 4 / 5`、smoothed rate edge massも
  `+0.000377954`悪化し、mechanism gateは3 / 7 PASSだった。rateの0方向収縮を
  除いてもpersistent TVT offsetへ安定転移しないため`stage0_fail_closed`。
  momentum / `sig_r` / sample / gate / blendをsame-OOF救済せず、Stage 1、
  inference、submissionなしでbranchを閉じ、完了済みのためbacklogから削除する。
- `exp425_symmetric_datum_reanchor_exact_hmm`はKaggle private CPU
  version 1（id_no `128930925`）でfixed32 baseline 32 + treatment 32 logical
  exact-HMMを`2,684.506175秒`、peak RSS `1.001877 GB`で完走した。parity、
  normalization、truth-late、finite、SHAはPASSしたが、full runtime投影は
  `64,847.602 > 30,600秒`でtechnical gateは12 / 13 PASS。soft datum方向一致は
  `0.396578 < 0.60`、passing foldは`1 / 5 < 4 / 5`、backward-cause SSE削減は
  `0.069758% < 10%`、matched-control reanchor massは`0.285635 > 0.10`で、
  mechanism gateは3 / 7 PASSだった。対称3枝とexact future evidenceでも
  absolute datum方向を識別できず、controlへもbranch massを割り当てた。
  trigger / prior / shift / readout / gateをsame-sample救済せず
  `stage0_fail_closed`。Stage 1、inference、submissionなしで閉じ、
  完了済みのためbacklogから削除する。同じtrigger / symmetric-datum scheduleを
  使う後続案はexp425をpositive evidenceとして扱わず、独立機構として再評価する。

- 2026-07-28、`exp427_affine_ar1_whitened_gr_likelihood_readout`はKaggle
  private CPU version 2（id_no `128931242`）で773 wells / 7,787 blocksを
  `4,358.768411秒`、peak RSS `1.264053 GB`で完走した。eligibleは
  697 wells / 5,615 blocksで、block率`0.721074 < 0.75`だけがtechnical FAIL。
  finite score / row identity coverage、13 candidates、outer-valid rho source
  overlap 0、`abs(rho)<=0.754092`、truth-late、Woodbury parity、runtime / RSSは
  PASSした。primary `affine_ar1`のMRR / top3は`0.386090 / 0.439181`で、
  matched identity-iid `0.388003 / 0.450401`、saved exp280
  `0.388146 / 0.449866`の双方を下回った。改善foldも両control比MRR `2/5`、
  top3 `1/5`。long-tailは両指標、hidden-like 2面はtop3、top1-regret p90もFAIL。
  shuffle比だけは5/5 foldsで正だったが、affine単独比MRR差は
  `+0.000486 < +0.005`、AR1単独比は`-0.000387`で複合効果を支持しない。
  `stage_0_failed_close_without_rescue`としてparameter / support / gate救済、
  HMM / PF / inference / submissionなしで閉じる。完了済みexp427と、その完全PASSを
  前提にした条件付きexp431はbacklogから削除する。

- 2026-07-28、`exp423_same_typewell_gr_dtw_truth_warp_transfer_readout`を
  Kaggle private CPU version 2 / 3 で完了した。same `native_overlap=1` group の
  outer-train donorだけを256-point GR-DTWで選び、top-5 donor truth-warpを
  query anchorへ転写した。primary RMSEは`14.103812714`でexp109 fixed
  `11.143366769`より`2.960445945 ft`悪化し、全5 foldsでnon-worse 0。
  top-5 per-well oracleも`12.285086482`で`1.141719713 ft`悪く、個別donor
  transferability headroom自体がなかった。top-1はstable randomより
  `1.233003803 ft`良かったが、pooled DTW cost-error Spearmanは
  `0.102226493 < 0.15`。supportは`286 / 773 wells`、score rows
  `0.368516928`で固定0.90 gateをFAILし、1000+、両hidden-like、by-well p95
  `+14.895650101 ft`、worst`+52.735848591 ft`もFAILした。query truth
  pre-freeze read 0、donor/query intersection 0、input SHA、path finiteはPASS。
  logical content SHA
  `6b5b54521ba6612665436f95d4ab3d42c711e8eb18a29bb2ad1916862849d3b3`
  は独立rerunで一致した。oracle不合格の事前分岐どおりtruth-warp transferを
  PF/Beam、inference、submissionへ昇格せず、parameter rescueなしで閉じた。
  exp423は完了済みのためbacklogから削除する。

- 2026-07-29、`exp435_tvt_memoryless_u_rate_dzonly_hmm`のKaggle private CPU
  Stage 0 version 1（id_no `129049294`）を`46.077013096 sec`、
  2 variants ×32 wells = 64 HMM well-runs、保存exp209 parent rerun 0、
  model / LightGBM / booster / PF / Beam / GPU各0で完了した。technical gateは
  finite、transition / posterior normalization、dz parity、truth / role-fold /
  episode pre-freeze read 0、SHA readback、runtime、RSSを全PASS。
  `memoryless_41rate` / `dz_only_r0`はforward-cause episode SSEを
  `27.205050% / 43.429062%`、persistent SSEを`11.244859% / 21.835503%`
  改善したが、改善wellは`4/16 / 5/16`、改善foldは双方`1/5`に留まった。
  matched-control pooled deltaは`+16.151527 / +13.705216 ft`、by-well p95は
  `+29.129905 / +24.955652 ft`と大幅悪化し、両variantともmechanism AND gateを
  FAILした。episode単位のhysteresis軽減signalはあってもTVT-only縮約の
  negative transferが支配的で、nonzero stationary supportもdz-onlyより安全ではない。
  `stage0_fail_closed_all_variants`としてStage 1 eligibleを空にし、rate重み /
  support / noise / emission / grid / gate / blend / selector救済、Stage 1、
  inference、submissionなしでbranchを閉じる。完了済みexp435をbacklogから削除し、
  今回のnegative resultだけに依存する後続候補は追加しない。既存優先順位は維持する。

- 2026-07-29、`exp438_u_state_fixed_lattice_exact_hmm`のKaggle private CPU
  Stage 0 version 1（id_no `129056676`）を、1 variant ×32 wells、
  保存exp209 parent rerun 0、ML / booster / PF / Beam / GPU各0で完了した。
  coordinate/emission/readout identity、constant-Z parity、brute-force、
  normalization、truth-late、SHA、RSSはPASSしたが、Stage 1 runtime投影は
  `33,907.307 sec > 30,600 sec`でFAILした。mechanismは7項目すべてFAILし、
  posterior-weighted quantization biasは`-43.580%`、forward-cause episode
  SSEは`-214.796%`、persistent episode SSEは`-824.234%`、改善well / foldは
  `2/16 / 0/5`だった。matched-control pooled deltaは`+43.320 ft`、
  by-well p95は`+72.481 ft`まで悪化した。連続座標contractが成立した状態で
  科学差分だけが強く悪化したため、fixed absolute-U latticeを
  `stage0_fail_closed`で棄却する。exp209は保存controlとして維持し、
  独立軸のexp435/436/437を再分類しない。Stage 1、inference、submission、
  grid / anchor / step / band / noise / rate / emission / blend / selector救済へ
  進まず、完了済みexp438をbacklogから削除する。今回のnegative resultだけに
  依存する後続候補は追加せず、既存優先順位を維持する。

- 2026-07-29、`exp439_continuous_kinematic_joint_transition_exact_hmm`の
  Kaggle private CPU Stage 0 version 1（id_no `129058811`）を実行した。
  fixed32の1 variant ×32 candidate HMM well-runs、保存exp209 parent rerun 0、
  ML / booster / PF / Beam / GPU各0の契約だったが、最初のwell `060ab2b8`の
  row 0、`source_rate=destination_rate=0`、
  `mean_shift=-0.11000000000021828 ft`で完了HMM well-run 0のまま停止した。
  固定0.35 ft lattice上でmeanを挟む`-0.35 / 0.0 ft`が許す最小分散
  `0.026400000000028373 ft^2`に対し、固定sigma `0.1225 ft`のtarget varianceは
  `0.015006249999999999 ft^2`しかなく、5/7/9-cellの非負分布ではmomentを
  同時保存できない。これは事前登録したtechnical fail-closeの再現であり、
  packageやsolverの不具合ではない。科学/mechanism gate、truth-late評価、
  prediction生成へ進まず、support / moment / noise / grid / rate / emission /
  prior / gate救済、再実行、Stage 1、inference、submissionなしで閉じる。
  完了済みexp439をbacklogから削除する。この失敗は固定grid/noiseとの
  representation incompatibilityを示すだけでexp209のpersistent stateや
  exp436/437/438を再分類しない。今回のnegative resultだけに依存する後続候補は
  追加せず、既存優先順位を維持する。

- 2026-07-29、`exp436_sparse_global_stratigraphic_potential`のKaggle private
  CPU Stage 0 version 2（id_no `129058940`）を完了した。1 primary candidate、
  6 formation report-only paths、fold 0最大36 sparse solves、保存exp226 control
  rerun 0、ML / booster / HMM / PF / Beam / GPU各0の契約で、input identity、
  source-valid overlap 0、target formation / GR / suffix truth read 0、duplicate 0、
  finite source、reported runtime`175.435738 sec`、RSS`0.549873 GB`はPASSした。
  ANCC / ASTNU / ASTNL / EGFDU / EGFDLは各fold 555–618 source contact wellsを
  持ち、fold 0の5面は30 sparse solvesを完了した。一方BUDAは全fold
  `5 / 4 / 4 / 5 / 6` wellsしかなく、固定最小32をFAILした。6面contractが
  揃わないためtarget queryを開始せず、runtime値はfull queryを含まない参考値として
  `stage0_fail_closed`でbranchを閉じる。
  version 1は同じ不足を例外終了したが、gateを緩めずversion 2で正常なfail-closeと
  manifest保存を検証した。formation除外 / contact定義 / support / aggregation /
  gate救済、再実行、Stage 1/2、inference、submissionは行わない。完了済みexp436を
  backlogから削除する。target-free supportだけでBUDAを事前除外する固定5面contractは
  exp436を再分類しない別仮説としてP3に置き、実装前に別実験・別承認を必須とする。

- 2026-07-30、`exp445_tvt_to_u_coordinate_parity_exact_hmm`のKaggle
  private CPU version 2（id_no `129095337`）を完了した。version 1はNumba
  初期化後のthread環境変数変更でHMM前に失敗し、科学contractを変えず
  `set_num_threads(1)`だけへ修正した。fixed32のcandidate 32 + paired parent
  32 = 64 HMM well-runsを`1,920.670 sec`、peak RSS `1.190 GiB`で完走し、
  technical gateは16/16 PASS。real position/rate posteriorとlog-likelihood差は
  すべて0、TVT mean/std最大差は`1.819e-12 ft`、truth/fold/role/episode/error
  readは0、artifact readback SHAもPASSした。したがって
  `U_t,j=P_j+Z_t`はexp209固定TVT格子の厳密なrow-shifted座標再ラベルであることを
  確認した。これは性能改善やexp438 fixed absolute-Uの救済ではなく、
  exp441--444の独立仮説も再分類しない。初回成功runだけではdeterministic
  anchorとせず、独立rerun、full OOF、inference、submissionへ進まない。
  完了済みexp445をbacklogから削除し、新規候補は追加せず既存優先順位を維持する。

- 2026-07-30、`exp458_acceleration_state_exact_runtime_engine_audit`の
  Kaggle private CPU version 2（id_no `129168013`）を完了した。1 scientific
  variant、1 runtime engine、fixed4×2 repeats、合計8 candidate HMM
  well-runs、parent/control/model/booster/PF/Beam/GPU各0の契約で、
  遅いrepeat`72.755703 sec`、exp444比`10.258353x`、fixed32/full投影
  `582.045625 / 14,114.606407 sec`、peak RSS`13.033188 GiB`、outer4、
  worker内thread 1、repeat SHA、leakage 0は全PASSした。一方、保存exp444比の
  prediction mean/std/acceleration posterior最大差
  `1.04135e-4 ft / 6.35657e-5 ft / 8.97726e-6`が固定閾値
  `1e-5 / 1e-5 / 1e-7`をFAILした。small dense誤差`3.64e-12 ft`だけでは
  長系列の累積数値差を保証できない。`stage0a_fail_closed`で閉じ、exp444を
  再分類せず、favorable rerun、gate/worker/thread/cache/precision/state/
  parameter救済、Stage 0B/1、inference、submissionを行わない。
  完了済みexp458をbacklogから削除する。次の原因検証は保存済みv2/exp444だけで
  最初の誤差増幅row/stateを特定するtarget-free studyに限定し、新規HMM runや
  exp458救済を行わない低優先P4候補として置く。

- 2026-07-30、`exp459_persistent_acceleration_state_likelihood_pf`のKaggle
  private CPU Stage 0 version 1（id_no `129167965`）を完了した。1 scientific
  variant、32 candidate PF well-runs、4,096 seed-well、2,048,000 particle
  starts、4 zero-acceleration sentinel wells、保存control/model/HMM/Beam/GPU
  rerun各0の契約で、technical gateは全PASSした。4 sentinelのprediction /
  log-likelihood / resampling / minimum ESS / clip countはexp404とbitwise一致し、
  最大誤差0、pre-freeze truth/control/role-fold/episode readも0だった。
  candidate runtime`928.287 sec`、full投影`22,423.933 sec`、RSS`0.795540 GiB`も
  上限内だった。一方、nonzero acceleration massは`0.666245`でも将来curvature
  方向一致は`0.501086`、positive fold`0/5`で、persistent episode SSEは
  `-11.6190%`悪化、matched-control pooled / p95は`+0.435213 / +1.785604 ft`
  悪化した。state collapseやruntimeではなく、固定persistent accelerationの
  GR識別力とcontrol安全性をmechanism gateで否定したと判断し、
  `stage0_fail_closed`で閉じる。exp444 / exp367を再分類せず、acceleration /
  transition / noise / particle / seed / temperature / emission / gate /
  blend / selector救済、Stage 1、inference、submissionを行わない。完了済み
  exp459をbacklogから削除し、新しいacceleration候補は追加せず、既存の独立した
  非acceleration P1/P2/P3候補を優先する。

- 2026-07-30、ユーザーがexp458 v2の微小数値差をStage 0B用途に限って
  明示許容したため、exp458のexact-parity FAILを変更せず後継
  `exp489_acceleration_state_fixed32_mechanism_audit`を作成し、Kaggle private
  CPU version 1（id_no `129171668`）を完了した。exp458 v2のfixed4をSHA検証して
  再利用し、残り28 HMM wellsだけをouter 4 / inner 1 threadで
  `794.097712 sec`、peak RSS `13.971542 GiB`で計算した。32 wells /
  156,088 rows、finite、artifact readback、pre-freeze leakage 0などtechnical
  `10/10`はPASS。nonzero acceleration massも`0.664839`でcollapseしていない。
  一方、future curvature方向一致は`0.500309`、positive fold `0/5`、
  forward-cause / persistent episode SSE削減は`0.4355% / -3.6667%`、
  persistent改善は`8/16 wells` / `2/5 folds`で6つのmechanism gateをFAILした。
  matched exp209 control safetyはpooled delta `-0.162849 ft`、by-well p95
  `+0.077808 ft`でPASSしたが、仮説対象のpersistent mechanismを救済しない。
  exp459のPF branchでも方向一致約50%・persistent悪化だったため、固定3状態
  persistent acceleration表現はexact-HMM/PFの両routeでnegative evidenceが
  揃ったと判断する。exp444/458の履歴を再分類せず、span/transition/prior/
  engine/gate救済、Stage 1、inference、submission、新規acceleration候補を行わず
  `stage0b_fail_closed`で閉じる。既存の非acceleration P1/P2/P3を優先する。

- 2026-07-30、`exp485_stratified_initial_rate_bank_pf`のKaggle private CPU
  Stage 0 version 1（id_no `129169067`）を完了した。1 scientific variant、
  32 PF well-runs、4,096 seed-well、2,048,000 particle starts、
  156,088 rows、control PF / model / booster / HMM / Beam / GPU rerun各0の
  契約で、5×100 allocation、interleave、rate/fallback、duplicate保持、
  finite coverage、posterior normalization、particle count、stable seed、
  exp404 duplicate-center bitwise parity、truth read 0、RSSをPASSした。
  25/32 wellsは複数center、7/32は単一center、fallback 0でglobal degeneracyは
  回避したが、component extinction seed fraction maxは`0.921875`だった。
  candidate runtimeは`1,278.942 sec`、full投影は`30,894.444 sec`となり、
  固定上限`30,600 sec`を`294.444 sec`（`0.962%`）超過した。14 checks中
  runtime projectionだけがFAILしたため`stage0_fail_closed`でbranchを閉じる。
  小幅超過でもgateを緩めず、favorable rerun、window/allocation/spread/
  particle/seed/temperature変更、Stage 1、inference、submissionを行わない。
  完了済みexp485をbacklogから削除し、この結果だけに依存する救済候補は追加せず、
  既存P1/P2候補を優先する。

- 同日、ユーザーはexp485の`30,894.444 sec`程度のfull runtimeを許容範囲と
  明示し、Stage 1実行を別途承認した。元の`30,600 sec` gate FAILは履歴として
  保持し、科学gateやPF設定を変更しないruntime例外として全773 wells CVへ進む。
  inferenceとsubmissionはStage 1結果後の別承認とする。

- 同日、`exp485_stratified_initial_rate_bank_pf`の全773 wells Stage 1を
  canonical kernel version 3で完了した。version 2で773 PF well-runs、
  98,944 seed-well、49,472,000 particle startsのtarget-free成果物をfreeze後、
  保存HMM integrity checkで停止したため、SHA固定private Datasetから
  candidate PF rerun 0でtruth-late評価だけを再開した。technical 19/19は
  PASSしたが、candidate `11.092618091`は保存exp404 `10.914522073`より
  `0.178096018 ft`悪く、positive foldは1/5、by-well p95 / worstは
  `+0.422388632 / +33.053515117 ft`、固定HMM+PF 50:50も
  `+0.032681136 ft`悪化した。high-missing scopeだけ`+0.018240364 ft`
  改善し、raw observed、raw missing、long-tail、hidden-like 2面は悪化した。
  一律equal-strata allocationが観測十分なwellでも親tail30 modeの有効粒子を
  減らす副作用がheadroomを上回ったnegative evidenceと判断する。元のStage 0
  runtime FAILは再分類せず、window/allocation/spread/particle/seed/
  temperature/gate/blend/selectorの同一OOF救済、新規initial-rate-bank派生、
  inference、submissionを行わず`stage1_gate_failed_terminal_close`で閉じる。
  この系統の新規backlogは追加せず、既存の独立したP1/P2候補を優先する。

- 2026-07-30、`exp486_exp226_geometry_residual_likelihood_pf`のKaggle
  private CPU Stage 0 version 1（id_no `129170320`）を完了した。
  2 scientific variants ×32 wells = 64 PF well-runs、8,192 seed-well、
  4,096,000 particle starts、156,088 rows、保存control PF / model /
  booster / HMM / Beam / GPU rerun各0の契約で完走した。exp226列allowlist、
  geometry coverage、variant式/state、finite prediction、common seed、
  execution count、両variant freeze前のtruth/control/role-fold read 0、
  RSS、geometry factor active、ESS、residual state non-degenerateはPASSした。
  一方、64 variant-wellの合計`7,487.545 sec`から求める事前固定full投影は
  `180,871.020 sec`となり、上限`30,600 sec`を大幅にFAILした。
  residual supportもmin/max
  `0.9999999999999988 / 1.0000000000000011`でstrict boundをFAILした。
  後者は正規化weightの浮動小数overshootだが、独立runtime FAILがあるため
  tolerance追加やgate式差し替えで救済しない。fixed32記述RMSEはabsolute
  unary `9.183489453`、slow residual state `10.399506240`、保存exp404
  `9.616740808`だったがCVではなく、winnerも選ばない。
  `stage0_fail_closed`でbranchを閉じ、parameter/noise/grid変更、二variant併用、
  favorable rerun、Stage 1、inference、submissionを行わない。完了済みexp486を
  backlogから削除し、この結果だけに依存する救済候補は追加しない。
  既存P1/P2候補の優先順位は維持する。

- 同日、ユーザーはexp486の実行時間を許容し、全773 wells Stage 1を
  明示承認した。元のruntime/support gate FAILは履歴として保持する。
  support strict-bound超過は最大約`1.1e-15`の正規化丸め誤差なので、
  Stage 1 technical readbackに限り`1e-12` toleranceを適用する。
  fixed32のabsolute unary記述値から候補を選ばず、absolute / residualの
  両方を`1,546` PF well-runs、`197,888` seed-well、`98,944,000`
  particle startsで実行し、保存exp404へ独立判定する。control PF / HMM /
  Beam / model / booster / GPU rerunは0。inferenceとsubmissionはStage 1
  結果後も別設計・別承認とする。

- exp486 Stage 1 version 2は両variant ×773 wells、3,783,989 rows、
  98,944,000 particle startsを完了してtarget-free predictionとmechanism
  ledgerをfreezeした後、exp209保存HMMの期待SHAが62文字になっていた
  manifest typoで`ERROR`になった。予測logical SHA
  `70a5ac662c9c58fe54d050f1350ed08e912ecb4edc6362e98e3c3663cd704ea8`
  と全raw/decompressed SHAを検証し、private Dataset
  `kentookumura/exp486-v2-stage1-frozen-targetfree`へ回収した。
  version 3は科学contractを変更せず、current PF rerun 0で同じtruth-late
  readoutだけを再開する。これはparameter/gate rescueではなく、freeze後の
  manifest typo recoveryとして扱う。

- exp486 Stage 1はcanonical private CPU kernel version 4で完了した。
  version 2でfreeze済みの1,546 PF well-runs、197,888 seed-well、
  98,944,000 particle startsをSHA固定Datasetからcurrent PF rerun 0で
  truth-late評価した。technical gateは全項目PASS。absolute geometry unaryは
  保存exp404 `10.914522073`に対して`9.726938029`（`+1.187584044 ft`）、
  4/5 folds、全事前scopeを改善し、固定HMM-PF 50:50も
  `+1.213888207 ft`改善した。一方、by-well delta p95 / worstが
  `+10.069321492 / +44.021977054 ft`で事前上限を大幅にFAILした。
  slow residual stateは`11.139812021`（`-0.225289948 ft`）、2/5 foldsで、
  raw observed、high missing、long-tail、hidden-like 2面とtailもFAILした。
  eligible variantは0。pooled改善だけでabsoluteをwinnerにせず、
  blend/selector/parameter/gate救済、inference、submissionを行わない
  `stage1_all_variants_gate_failed_terminal_close`とする。このnegative evidence
  だけに依存する新規backlogは追加せず、既存P1/P2候補を優先する。

- 2026-07-30--31、`exp490_geometry_centered_mean_reverting_offset_hmm`を完了した。
  Stage 0 version 1（id_no `129180511`）はpersistent episode SSE`69.893385%`削減、
  13/16 wells、5/5 folds、matched-control pooled`-0.462223 ft`を得たが、control
  by-well p95`+3.118472 ft`とruntime投影`51,464.889 sec`をFAILした。その判定を
  維持した明示overrideで、同じ1 variantを4 target-free shard、合計773 HMM
  well-runs、保存control再decode / model / booster / PF / Beam / GPU各0でfull OOFした。
  strict merge version 1（id_no `129321382`）のCVは`8.480155260`で、保存exp357
  `9.737195157`から`1.257039898 ft`、exp226 final`9.427109597`から
  `0.946954337 ft`改善した。4/5 folds、MD 1000+、hidden-like 2面が改善し、
  persistent episode SSE`41.409965%`削減、episode count`-59`、recovery 256/512
  `+0.036050/+0.025078`もPASSした。一方、449 wells改善 / 324悪化で、by-well
  p95`+7.257814 ft`、worst`+49.602560 ft`をFAIL。14 gate中12 PASSの
  `stage_1_full_oof_failed_closed`とした。その後の明示LB監査overrideで公開test
  3 wells / 14,151 rowsを推論し、technical 13/13、submit-check FAIL/WARN 0を確認して
  submission ref `55163886`を送信したが、hidden再実行は未処理例外でscoreなし。
  原因は公開sample SHA・行数・well数を固定したruntime guardのhidden非互換であり、
  2026-08-02にruntime test全件を開始時に走査し、sample / horizontal / typewell well集合と
  全unknown row IDを照合してrows / wellsを動的導出するinference version 2を実装した。
  scientific contractとtrain結果は不変。同じcanonical private CPU kernel version 2を
  実行し、14,151 rows / 3 wells、technical 14/14、submit-check FAIL/WARN 0でCOMPLETE。
  v1/v2 public submissionはbyte-identicalだった。ユーザー明示承認によるversion 2
  submission ref `55180208`はhidden再実行を通過してPublic LB `9.680`。exp226 direct
  `9.837`より`-0.157`改善した一方、direct exact HMM `9.063`より`+0.617`、self-GR HMM
  `9.318`より`+0.362`、direct LikPF `8.797`より`+0.883`悪く、CVからも
  `+1.199845`乖離した。runtime修正の成功とscientific modelの競争力を分離し、
  physical-route LB anchorへは昇格しない。fail-closeを維持し、復元力自体のpositive
  evidenceとfixed-strengthのtail riskは保存full OOFの原因readoutへ引き継ぐ。同モデルの
  half-life / noise / grid / gate救済や追加提出は行わず、既存P1/P2候補の優先度を維持する。

- 2026-07-30、`exp483_huber_gr_filtering_likelihood_pf`のKaggle private CPU
  Stage 1 version 2（id_no `129169339`）を全773 wells / 3,783,989 rowsで
  完了した。1 scientific variant、773 candidate PF well-runs、98,944
  seed-well、49,472,000 particle starts、保存control PF / HMM / Beam /
  LightGBM / booster / GPU rerun各0の契約、raw identity、finite coverage、
  truth-late freeze、保存control/blend parity、SHA、runtime/RSSは全PASSした。
  一方、Huber candidate `11.095404595`は保存exp404 `10.914522073`より
  `0.180882522 ft`悪化し、改善foldは3/5。raw observed `-0.253381263 ft`、
  1000+ `-0.208228055 ft`、hidden-like typewell-purged `-0.109562968 ft`、
  by-well p95 `+0.520909635 ft`、worst `70e1788b +33.458522531 ft`、
  fixed HMM/PF 50:50 `+0.077245509 ft`で科学AND gateをFAILした。
  `terminal_close_without_huber_or_pf_rescue`として、delta/scale/temperature/
  clip/mixture、particle/seed/dynamics、well/row gate、blend/selector、
  same-OOF rescue、inference、submissionを行わない。完了済みexp483を
  backlogから削除し、この結果だけに依存する救済候補は追加せず、既存候補を
  優先する。

- 2026-07-30、`exp484_student_t_gr_filtering_likelihood_pf`のKaggle private
  CPU Stage 1 version 3（id_no `129170461`）を全773 wells /
  3,783,989 rowsで完了した。1 scientific variant、773 candidate PF
  well-runs、98,944 seed-well、49,472,000 particle starts、保存control PF /
  HMM / Beam / LightGBM / booster / GPU rerun各0の契約、formula、
  stable seed、ESS/resampling、raw identity、finite coverage、truth-late
  freeze、保存control/blend parity、SHA、runtime/RSSは18/18 PASSした。
  一方、Student-t candidate `10.897096923`の保存exp404 `10.914521913`比改善は
  `+0.017424990 ft`で必要`+0.05 ft`未満、改善foldは2/5。raw observed
  `-0.068900357 ft`、hidden-like typewell-purged `-0.130146256 ft`、
  by-well p95 `+1.455066656 ft`、worst `d924e971 +16.664889733 ft`で
  科学AND gateをFAILした。raw missing、高missing、1000+、spatial、
  fixed HMM/PF 50:50は改善したが、primary FAILを救済しない。
  `terminal_close_without_student_t_or_pf_rescue`として、df/scale/temperature/
  clip/mixture、particle/seed/dynamics、well/row gate、blend/selector、
  same-OOF rescue、inference、submissionを行わない。完了済みexp484を
  backlogから削除し、この結果だけに依存する救済候補は追加しない。

- 2026-07-31、`exp491_exp226_final_tvt_rate_direct_hmm`のKaggle private CPU
  Stage 0 version 2（id_no `129213586`）をfixed32 / 156,088 rowsで完了した。
  1 scientific variant、32 candidate HMM well-runs、control / ML / booster /
  PF / Beam / GPU各0の契約、strict allowlist、truth/role/episode-late freeze、
  first-difference / rate identity、normalization、SHA readback、runtime / RSSの
  technical gateは全件PASSした。all32はexp226 final `7.976056519`に対して
  candidate `12.290250882`（`+4.314194362 ft`悪化）、persistentは
  `8.757067232 → 16.169236485 ft`、改善foldは3/5、episode SSE reductionは
  `-3.142299927`、by-well p95は`+22.805438506 ft`だった。matched-control
  safetyだけPASSし、mechanism gate 6/7をFAILしたため`stage0_fail_closed`とする。
  Stage 1、same-OOF rate / emission / grid / blend / selector調整、PF救済、
  inference、submissionを行わない。完了済みexp491と、その成功を前提にした
  条件付きPF後続案をbacklogから削除する。この失敗だけに依存する救済候補は
  追加せず、既存候補を優先する。

- 2026-07-31、`exp494_exp413_cat_xgb_physics_bounded_stack`のKaggle private
  T4 train version 5（id_no `129213293`）を3,783,989 rows / 773 wells /
  final370で完了した。CatBoost 5 + XGBoost 5の10/10 modelsを学習し、
  保存exp413 LightGBM、selector、PF/HMM/Beamの再学習は0。CatBoost /
  XGBoost / fixed physics単体はexp413比`+0.223223 / +0.167667 /
  +0.185416 ft`悪化したが、固定bounded stackは`7.884802794`から
  `7.827450885`へ`0.057351909 ft`改善し、5/5 folds、near / mid /
  far、hidden-like 2面を全て改善した。一方、by-well p95
  `+0.634420635 ft`、worst`+3.843640672 ft`で固定tail gateをFAILした。
  `train_complete_guard_failed_closed`としてexp413をanchorに維持し、
  conditional gate、weight/candidate/parameter/bound/threshold rescueは行わない。
  後のユーザー明示overrideで未調整constant stackだけをhidden-safe参考提出し、
  ref `55134873`は268分後にCOMPLETE / Public LB `7.228`となった。exp413
  `7.201`比`+0.027`悪化のためexp494は不採用、overall / ML anchorはexp413を
  維持する。一方、旧exp082 `7.601`を上回るためensemble-routeのLB referenceだけを
  exp494へ更新する。完了済みexp494をbacklogから削除する。
  平均改善とtail悪化の併存原因は、保存済みOOF / by-well / weightだけを使う
  0-model attribution readoutをP4へ追加するが、現行P1/P2候補を追い越させない。

- 2026-07-31、`exp495_uncertainty_weighted_exp226_rate_observation_hmm`のKaggle
  private CPU Stage 0A version 1（id_no `129285050`）を1 diagnostic variant、
  773 wells / 3,783,989 suffix rows / 5 folds、HMM / model / booster / PF /
  Beam / GPU / control再実行各0で完了した。strict exp226 allowlist、truth-late
  freeze、missing / duplicate 0、formula parity 0.0、uncertainty coverage 1.0を含む
  technical gateは全件PASS。low-sigma schedule gain `8.843818%`、schedule改善
  4/5 folds、Spearman正方向5/5 folds、fallback 0/773もPASSした。一方、prefix
  MADとsuffix absolute geometry-rate errorのpooled Spearmanは`0.088435 < 0.20`、
  low/high-sigma rate RMSE gainは`7.645442% < 10%`でmechanism gateをFAILした。
  prefix MADは弱い方向性を持つがsuffix信頼度を十分に分離しないため、事前登録した
  `close_before_hmm_implementation_without_window_sigma_scale_or_gate_rescue`に従い、
  当初はHMM実装前に閉じた。その後ユーザーの明示overrideにより、事前登録条件を
  変更せずStage 0B fixed32をprivate CPU version 4で実行した。1 variant / 32 HMM
  well-runs / 156,088 rows、親control再実行・model・booster・PF・Beam・GPU各0。
  candidate all32 RMSE `13.069257`はsaved exp355 `10.677951`より`+2.391305 ft`、
  persistentは`+2.143391 ft`、matched controlはsaved exp209比`+5.026402 ft`悪化した。
  改善foldは2/5、episode SSE reduction `6.9667% < 10%`、by-well p95
  `+16.564282 ft`、worst`+23.911032 ft`でmechanism 7/7 FAIL。technicalもposterior
  normalization `5.6292e-06 > 1e-06`の1件をFAILした。uniform-factor parent parity、
  transition row sum、finite coverage、runtime、RSS、truth-late guardはPASS。
  `close_without_sigma_window_scale_temperature_emission_grid_blend_selector_or_pf_rescue`
  としてStage 1 / inference / submissionなしでterminal closeする。完了済みexp495を
  backlogから外した状態を維持し、保存Stage 0A/0B artifactだけの原因分解をP4に残す。

- 2026-07-31、`exp452_scale5_likpf_direct_public_lb_audit`はユーザー承認範囲内の
  private CPU inference version 1（id_no `129271895`）を完走した。固定
  `likpf_scale_5_x1p0`を14,151 rows / 3 wells、500 particles × 128 seedsで生成し、
  fallback 0、exp413 v4公開surfaceとのfloat32最大差`0.0 ft`、logical content SHA
  `b713ade7...`を確認した。取得した`submission.csv`はsampleのheader・行数・ID順序と
  完全一致し、submit-checkはFAIL 0 / WARN 0でPASSした。Codexはcompetition
  submissionを行わなかった。その後ユーザーが外部提出し、2026-08-01にref
  `55149125`がexp452であることを明示確認した。Kaggle statusは`COMPLETE`、Public LB
  `8.797`。同じSHA256 seed familyのexp434 v10 arithmetic LikPF `9.807`より`1.010`
  改善し、OOF `0.680375810 ft`改善と方向一致した。一方、exp417のby-well p95
  `+2.941688483 ft` / worst `+25.311274575 ft` FAILは維持する。公開splitの記述census
  として閉じ、追加run、rerun、再提出、自動昇格、LB適応は行わない。この単発LBから
  新しいtemperature/seed候補を追加せず、事前凍結済みexp453/exp454の順序も変更しない。

- 2026-08-01、`exp498_geometry_mean_reversion_tail_regime_physics_readout`のKaggle
  private CPU version 2（id_no `129328553`）を3,783,989 rows / 773 wells、
  readout 1 / truth-late 5 folds、HMM / prediction / model / booster / PF / Beam /
  GPU各0で完了した。fixed input SHA、horizontal suffix truth read 0、pre-freeze
  outcome read 0、feature / bucket / identityを含むtechnical checksは全PASS。
  一方、事前固定した`weak observation AND geometry disagreement>=10 ft AND early
  abs offset>=5 ft`は0 wellsだった。weak observationは359 wellsだが、geometry
  conflictは0 wells（最大`5.337991 ft`）、material early offsetは1 wellだけで、
  supported fold 0/5、catastrophic 51 wellsのcapture 0、physics checks 6/6 FAIL。
  `terminate_mean_reversion_tail_regime_cause_tracking`として、threshold / bucket /
  interaction / same-OOF救済、復元力を弱める後続式、inference、submissionを行わない。
  exp490のterminal closeを維持し、完了済みexp498をbacklogから削除する。このFAILだけに
  依存する後続候補は追加せず、既存候補の優先度も変更しない。

- 2026-08-01、`exp499_exp490_cross_fitted_well_application_selector`のKaggle
  private CPU version 2（id_no `129362815`）を完了した。保存済みexp490 / exp357
  full OOFとexp498物理特徴から、truth前に32 target-free well特徴をSHA freezeし、
  outer 5 × inner 4でweighted Ridge / shallow HGB / always-exp490 safeguardを比較した。
  1 variant / 2 learned configs / 45/45 CPU fits、new prediction / HMM / PF / Beam /
  LightGBM booster / GPU / control再学習各0で、identity、SHA、phase分離、finite、outer
  coverageを含むtechnical checksは全PASSした。一方、cross-fitted scoreはpooled AUC
  `0.521151`、AUC>=0.55は`1/5 folds`でpredictability gateをFAIL。policy RMSE
  `8.514310626`はalways-exp490 `8.480155260`より`0.034155367 ft`悪化した。
  4 foldsはalways-exp490を選び、唯一HGBを採用したfold 1も`8.659383 -> 8.822361`
  へ悪化。716/773 wellsへexp490を適用し、harmful 300、catastrophic 48を残し、
  selected-minus-exp357 p95 / worstは`+7.098191 / +49.602560 ft`でsafe-router gateも
  FAILした。report-only oracle `6.560582422`と最強単一特徴
  `mean abs(exp357-exp226)` AUC `0.591912`は選択余地と弱いranking signalを示すが、
  未知wellへ一般化するhard routing根拠にはならない。
  `close_safe_target_free_exp490_router_without_same_oof_rescue`としてthreshold / feature /
  model追加、inference、submissionを行わず、exp490 terminal closeを維持する。この結果
  だけに依存するselector救済backlogは追加しない。exp500の固定PF機構移植は別仮説・
  別承認のまま維持し、exp499 signalをadaptive gateへ持ち込まない。

- 2026-07-31、`exp434_physics_candidate_public_lb_audit`は凍結した全12候補の
  Public LB censusを完了した。batch 3は`pf_ancc=12.061`（ref `55133068`）、
  `beam_mean=15.563`（`55133072`）、SHA256 seed版`likpf_mean=9.807`
  （`55133074`）で全件COMPLETE。全体首位はK16 + exact HMM `7.678`、固定
  3-way `7.800`、K16 + self-GR HMM `7.913`が続き、OOF/LB Spearman順位相関は
  `0.846154`だった。primitiveは`exact_hmm 9.063 < selfgr_hmm 9.318 <
  likpf 9.807 < K16 9.837 < pf_ancc 12.061 < beam 15.563`のLB順で、K16は
  OOF 5位からLB 10位へ逆転した。結果は記述的な比較根拠に限定し、weight tuning、
  candidate追加、train-side/最終提出への自動昇格は行わない。完了済みexp434を
  backlogから削除する。

- 2026-08-01、`exp500_exp490_mean_reversion_residual_likelihood_pf`のKaggle private
  CPU Stage 0 fixed44をversion 2（id_no `129380054`）で完了した。1 variant / 44 PF
  well-runs / 5,632 seed-well / 2,816,000 particle starts、control PF / HMM / Beam /
  model / GPU再実行0で、leakage、K16 half-life、exp486 parity、有限性、runtime、RSSを含む
  technical checksは13/13 PASS。persistent subsetはexp486比SSE `50.941239%`削減、
  13/16 wells・5/5 folds改善した。一方matched-control pooled / by-well p95はexp404比
  `+1.079414 / +7.468536 ft`、PF sentinel worst-wellは`+9.159571 ft`で3 safety gateを
  FAILした。固定mean-reversionの効果はPFでも非一様であり、exp490のpooled改善 / tail悪化を
  解消しない。`terminal_close_without_same_fixed44_rescue`としてStage 1、inference、
  submission、adaptive gate、parameter grid、blend / selector救済へ進まない。完了済みexp500を
  backlogから削除し、保存診断だけのharm attributionをP4候補として追加する。

- 2026-08-02、ユーザーの明示overrideにより、上記Stage 0 FAILを再分類せず、exp500の
  変更なし単一variantをKaggle private CPU 4 shards + merge version 3でfull OOF実行した。
  773 wells / 3,783,989 rows / 98,944 seed-well / 49,472,000 particle starts、control再実行0で、
  technicalは18/18 PASS。candidate RMSE `8.813504627`はexp404 `10.914522073`から
  `2.101017446 ft`改善し、5/5 folds、全6固定scope、exp408 / exp410 episodeを改善した。
  しかしby-well delta p95は`+6.653601019 ft`、worst wellは`+46.154671032 ft`悪化し、
  固定tail gate 2件をFAILした。`stage1_fail_closed_under_override`として、平均改善だけで
  安全性FAILを上書きせず、same-OOF rescue、inference、submissionなしで終端閉鎖する。
  P4原因readoutはfixed44限定から保存full OOF tail artifact中心へ更新する。

- 2026-08-02、`exp503_exp490_strength_weakness_prefix_policy_readout`のKaggle
  private CPU version 3（id_no `129477630`）を3,783,989 rows / 773 wells、保存済み
  exp490 / exp357 OOFとexp499 target-free 32 features、29 fixed fade profiles、outer 5、
  prefix/context depth-2 tree各5 fits、descriptive KMeans 1 fitで完了した。control再学習、
  new prediction、HMM/PF/Beam、LightGBM、GPU、inference、submissionは全て0。technicalは
  PASS。exp490は449 wells改善 / 324悪化だが、positive gainの36.9231%を上位10 wells、
  negative harmの57.8542%をworst10 wellsが占めた。truth-aware correction alignmentと
  benefitのSpearmanは`0.820838`で、weak/strong medianは`-0.214/0.757`。exp357 RMSE
  上位quintileだけexp490が`-6.992134 ft`改善し、下位4 quintilesは全てpooled悪化した。
  0--512 suffix rowsも悪化し、1024+で改善が拡大するため、exp490はexp357のpersistent
  whole-well biasを救う少数well型の補正と判断する。最強target-free signal
  `parent_exp226_abs_mean`もAUC`0.591912`でexp499のrouter FAILと整合する。
  公開`fle3n-rogii-v5`型fadeのtau85は`-0.000502 ft`で実質ゼロ。grid内tau500は
  `8.480155260→8.447032560`、5/5 folds、exp490比by-well p95/worst
  `+0.080156/+1.195616 ft`だったが、version 2後に選んだ探索的characterizationであり、
  exp490自体のparent比tailは残る。outer strong fadeは`8.098662373`、prefix treeは
  `7.911444631`へ平均改善した一方、fold 3を`+0.903082/+0.620514 ft`悪化させ、
  prefix tree p95/worstも`+3.458136/+20.766347 ft`。平均gate PASSをtail-safeに
  再分類せずinferenceへ進めない。early truth 128/256は後半を悪化し、512でもgain
  `0.021980 ft` / transfer Spearman`0.150234`だけなので、masked-prefix HMM replay
  triggerはFAIL。exp503を完了済みとしてbacklogへ残さず、adaptive prefix、hard router、
  cutoff replay、standalone submitを閉じる。tau500は将来exp490予測を別目的で再利用する
  場合の低優先fixed regularizer evidenceに限定し、現行P1/P2を追い越さない。

- 2026-08-02、`exp502_exp501_fixed13_selector_replacement_on_exp413`のKaggle private T4
  version 1（id_no `129459588`）をreplacement 1 variant × LightGBM 3 configs × outer 5 =
  15 / 15 modelsで完了した。saved exp413 control、exp501 selector、exp413 signed selectorの
  再学習とHMM/PF/Beam再実行は0、technical checksは全PASS。exp502 RMSE
  `7.882143903`はexp413 `7.884802794`から`0.002658891 ft`だけ改善し、必要な
  `0.03 ft`を未達した。fold 0--2は改善したがfold 3 / 4は
  `+0.116026853 / +0.234685837 ft`悪化し、hidden-like spatial / typewell-purgedも
  `+0.139586563 / +0.140943998 ft`悪化して固定scope上限`+0.02 ft`をFAILした。
  selector-level改善はexp413 downstreamへ安定転移せず、report-only tailもp95 / worst
  `+1.293097772 / +8.159899027 ft`。`FAIL_CLOSE_EXP501_FIXED13_SELECTOR_REPLACEMENT_ON_EXP413`
  としてsame-OOF救済、inference、submissionなしで閉じ、完了済みexp502をbacklogから削除する。

- 2026-08-02、`exp504_h512_regret_weighted_block_rank_selector`のKaggle private CPU
  version 1（id_no `129488458`）を1 scientific variant / 1 config / outer 5 / 5 CPU
  boostersで完了した。親control再学習、candidate再生成、PF/HMM/Beam、GPU、inference、
  submissionは全0で、technical gateは全PASS。H512 block rankのpooled RMSE
  `8.114276980`はfixed anchor `8.238331546`から`0.124054566 ft`改善したが、nonworseは
  `3/5 folds`。hidden-like spatial / typewell-purgedは`+0.285759 / +0.269833 ft`、
  by-well delta p95 / worstは`+2.963656 / +16.799044 ft`で固定tail gateをFAILした。
  pair accuracy `0.741908`に対してH512 exact top-1は`0.112624`、anchor選択率
  `0.377809`で、平均的なpair判別はpooled gainへ転じてもfold 3/4と分布外寄りwellの
  hard choice safetyへ転移しなかった。`FAIL_TERMINAL_CLOSE_WITHOUT_HORIZON_LOSS_WEIGHT_OR_THRESHOLD_RESCUE`
  として、horizon/loss/weight/model/threshold/guard救済、再実行、inference、submissionなしで
  終端閉鎖し、完了済みexp504をbacklogから削除する。原因確認は保存生成物だけを使う
  0-model / 0-prediction readoutを低優先P4へ置き、現行P1/P2/P3を追い越さない。

- 2026-08-03、`exp505_exp490_tau500_fade_fixed13_on_exp413`のKaggle private CPU
  version 1（id_no `129519165`）を1 variant / 2 objectives / outer 5 × inner 4 =
  40 / 40 modelsで完了した。control再学習、HMM/PF/Beam、GPU、inference、submissionは全0。
  exact 8列allowlist、raw/decompressed SHA、global key / suffix / `md_since` parity、source fold
  不使用、strict nested leakage、40 model / 25 compact partitionを含むtechnical checksは全PASS。
  tau500 fade fixed13 hard OOF `8.243315437`はraw exp501 `8.264890209`から
  `0.021574771 ft`改善し、4/5 folds、固定7 scope、fade top1 pooled `55.2414%` / 5/5 foldsを
  PASSした。しかしfixed12比by-well p95縮小は`0.000036536 ft < 0.10 ft`、worst縮小は
  `0.173168079 ft < 1.0 ft`で、material tail 2条件をFAILした。固定fadeは平均を改善しても
  exp501のtailを安全化しないため、`FAIL_CLOSE_WITHOUT_STAGE_D_OR_SAME_OOF_RESCUE`として
  Stage D、same-OOF救済、inference、submissionなしで終端閉鎖する。完了済みexp505を
  backlogから削除し、fade利用55%でもtailがほぼ不変だった原因確認だけを0-model / saved-
  artifact-onlyのP4候補へ置く。

- 2026-08-03、`exp497_strict_public_core_fold_safe_ensemble_on_exp413`のStage P/M/Eを
  Kaggle CPU/GPUで完了した。Public-LB固有overlayを除いたstrict public-coreを、1 variant、
  LGB120 + Cat80 = 200 boosters、Ridge10、exp413再学習0でouter 5 / inner 4 OOF化した。
  Stage E meta5 cross-fit blendはexp413 `7.884802794`から`7.874488150`へ`0.010314644 ft`
  改善し、5 weightは全て正かつ0.30未満だった。一方fold 0 / 4は`+0.025357 / +0.139179 ft`、
  hidden-like 2面は`+0.105138 / +0.097410 ft`、by-well p95 / worstは
  `+0.700720 / +7.541588 ft`悪化し、事前固定all-AND gateをFAILした。
  `completed_gate_failed_closed`としてexp413をselected anchorに維持し、same-OOF救済、
  submissionなしで科学評価を終端閉鎖した。後続の明示承認によるStage I v4ではcurrent-test用
  LGB24 + Cat16の40 boosterとRidge 2をKaggle GPUで保存し、335,918,672 bytes、model-set SHA
  `dcc2166f...626`、全file SHA/bytes、一意path、reload parity最大差0.0、14,151行prediction契約を
  PASSした。exp413再学習・再推論とsubmissionは0。後続承認でmodel読込型hidden-test inference
  候補も実装し、exp497/exp413学習0、dynamic exp413、fixed blend、29 testsと実model artifact契約を
  PASSした。後続実行指示により正規Notebookへ採用し、76-file package SHA readbackとT4 metadataを
  PASSしてprivate Kaggle version 1（id_no `129666751`）をpushした。version 1はdynamic exp413、
  strict特徴、全保存model推論後、visible parityがstrict `0.001281` / blend `0.014195 ft`で共通許容
  `0.001`を超えて`ERROR`。OOM・入力欠落・model破損ではなく外部submitも0。後続承認でstrict/blend
  toleranceを`0.002 / 0.020 ft`へ分離し、parent-only中間submissionを隔離するversion 2を30 tests、
  76-file/remote marker readback後に同じprivate T4で完了した。exp497 40 + Ridge 2、exp413 75をloadし、
  fit 0。strict/blend parityは`0.001281 / 0.014195 <= 0.002 / 0.020 ft`でPASSした。final
  `submission.csv`は14,151行、sample ID順、重複・欠損・finite、serialized blend差0.0、SHA、
  submit-checkを全PASS。外部competition submitは0で、科学gate FAILとselected anchor exp413は維持する。
  完了済みexp497をbacklogから削除し、
  保存成果物だけのtail/correlation原因分解をP4へ置く。これによりexp506の事前anchorは
  exp413に確定し、別承認後の0-model CPU監査として現行P1へ繰り上げる。

- 2026-08-03、`exp507_exp504_nested_rank_compact_addonly_on_exp413`のStage N / Dを完了した。
  Stage N CPU v1（id_no `129565024`）は20 rank models / 25 strict nested partitions、held
  outer/inner overlap 0、forbidden feature 0を含むtechnical 8/8をPASS。Stage D private T4 v1
  （id_no `129584313`）は`final370 + rank45 = final415`、1 treatment × 3 configs × 5 folds =
  15 models、保存exp413 control再学習0を完走しtechnical PASSした。しかしexp507 / exp413
  pooled RMSEは`7.889515566 / 7.884802794`、gain`-0.004712771 ft`、nonworse folds`2/5`、
  最大scope delta`+0.036938807 ft`で固定性能3条件をすべてFAILした。fold 0 / 1 / 4と
  short / hidden-like scopeの悪化を、fold 2 / 3の改善で相殺できなかった。
  `FAIL_CLOSE_WITHOUT_PAIR_FEATURE_SUBSET_TEMPERATURE_OR_GATE_RESCUE`としてsame-OOF救済、
  inference、submissionなしで終端閉鎖し、完了済みexp507をbacklogから削除する。必要なら
  saved-artifact-only / 0-modelのdownstream transfer原因readoutだけを低優先P4で検討する。

- 2026-08-04、`exp508_exp413_public_trajectory_postprocess_audit`のKaggle private CPU
  version 1（id_no `129625989`）を完了した。selectable primaryは保存exp413 Stage D OOF
  最終TVTへのwell別Savitzky--Golay `window=61 / polyorder=3`の1本、tau85単独とtau85+SGは
  primary decision freeze後のreport-only。1 primary / 2 controls、model / booster / PF/HMM/Beam /
  GPU / 親再学習各0で、technical / leakage / SHA checksを全PASSした。SGはexp413
  `7.884802794→7.878669067`、gain `0.006133728 ft`、5/5 folds、固定5 scopeを全改善し、
  by-well p95 / worst deltaも`-0.001344491 / -0.000417966 ft`、first-row correction p95 / maxも
  `0.289606691 / 0.810694404 ft`で安全性条件を全PASSした。しかし事前固定したpooled gain
  `>=0.01 ft`だけを未達としたため、all-AND gateをFAILした。
  `FAIL_CLOSE_WITHOUT_SG_GRID_WARMUP_ROUTER_OR_GATE_RESCUE`として、gate緩和、report-only救済、
  well router、inference、submissionなしで終端閉鎖する。完了済みexp508と、exp508 PASSを
  前提にした条件付きrouterをbacklogから削除し、独立P1のexp506を次候補として維持する。

- 2026-08-04、`exp506_exp490_mean_reversion_correction_blend_on_exp413`のKaggle private CPU
  version 2（id_no `129631767`）を3,783,989 rows / 773 wells、1 primary / 1 report-only control、
  outer/meta 5/5、model / booster / HMM/PF/Beam / GPU / 親再学習各0で完了した。technical、leakage、
  input/SHA checksは全PASSしたが、primary `anchor + lambda*(exp490-exp357)`はexp413
  `7.884802794→7.902068462`と`0.017265668 ft`悪化した。lambdaは
  `[0, 0.041578388, 0, 0.004513714, 0]`、deployment中央値`0.0`、nonworse`3/5 folds`、
  固定scope`0/5`、by-well p95 / worst`+0.054729 / +1.816050 ft`で、pooled、fold、scope、
  worst-well、lambda-positive gateをFAILした。`FAIL_CLOSE_WITHOUT_WEIGHT_SCOPE_COMPONENT_OR_GATE_RESCUE`
  として終端し、exp413を維持、inference / submissionなし。完了済みexp506をbacklogから削除し、
  report-only固定10% convex controlの推論候補化、weight再fit、exp506 gate再評価は禁止する。

- 2026-08-04、ユーザー依頼によりexp490の予測・アイデア・アルゴリズム再利用を横断調査し、
  `docs/surveys/exp490_reuse_strategy_20260804.md`へ記録した。保存exp506 OOFから固定10%
  `0.90*exp413+0.10*exp490`を再構成するとCVは`7.884803010→7.734534349`、5/5 folds、
  MD 3面、hidden-like 2面を全改善した一方、by-well p95 / worstは
  `+0.549195 / +2.657049 ft`、86 wellsが`+0.25 ft`超悪化した。1%まで縮めても
  CV gain`0.017009 ft`に対しworst`+0.253736 ft`で、post-hoc micro-blendも非選択のまま閉じる。
  exp413 / exp490 Public LB `7.201 / 9.680`では10% blendがLB非悪化となるhidden residual
  correlation上限は`0.710552`、OOF実測は`0.726676`であり、同相関を仮定したLB推定は
  約`7.215`。実測scoreではないが提出候補化を支持しない。exp499既存32 target-free featuresへ
  6 primitive physical candidatesのexp490 correction方向合意7特徴を加えたouter-fold Logisticは
  beneficial-well AUCを`0.659944→0.671898`、fold最低を`0.625965→0.646276`へ改善したが、
  hard applyはp95 / worst`+1.138018 / +49.602560 ft`。exp413固定10%の`>+1 ft` harm AUCも
  `0.552255`と不安定で、二頭hard gateはalways-fixed10より悪くworstを残した。したがって
  hard router再試行ではなく、K16 segment model-evidence readout、small add-only mechanism/risk
  features、position/rate factorial、evidence PASS後だけのswitching dynamicsを未着手候補へ追加する。
  exp509/510の最終提出優先順位は変更しない。
