---
title: ROGII実験横断メモ履歴
date: 2026-08-09
types:
  - comparison
experiments: []
topics:
  - history
  - experiment_summary
status: final
summary: "旧experiment_summary.mdから退避した、主な発見と変更履歴の手書き記録。"
---

# ROGII 実験横断メモ履歴（2026-08-09退避）

- 対応する上位仮説: なし

この文書は旧 experiment_summary.md の手書き部分です。現在の比較表はルートの experiment_summary.md、数値は各実験の metrics.json、解釈と採否判断は result.md を正とします。

## 主な発見

- exp512_hjyact_v2_final_10pct_hedge_on_exp413 のユーザー提出ref `55255459`は
  `COMPLETE` / Public LB `6.541`。固定0.50/0.50 blendはexp413 / exp510 `7.201`を
  `0.660`、source公開値`6.568`を`0.027`改善し、新しいoverall / ensemble Public-LB anchorとなった。
  数値score付きCOMPLETEにより提出hidden rerunのplatform上限内完走は確認できたが、正確なhidden runtime、
  honest OOF、Private LB、hidden stochastic determinismは未確認で、private-generalization採用とは分ける。

- exp508_exp413_public_trajectory_postprocess_audit はKaggle private CPU version 1
  （id_no `129625989`）を1 selectable primary / 2 report-only、model / booster / HMM / PF /
  Beam / GPU / parent再学習各0で完了した。固定SG61/p3は保存exp413 OOF
  `7.884802794→7.878669067`、gain `0.006133728 ft`で5/5 foldsと固定5 scopeを全改善。
  by-well p95 / worst delta `-0.001344491 / -0.000417966 ft`、first-row correction p95 / max
  `0.289606691 / 0.810694404 ft`、technical checksも全PASSした。しかし事前固定した
  pooled gain `>=0.01 ft`だけを未達としたため、all-AND gateをFAILし、report-only救済、
  router、inference、submissionなしで終端閉鎖した。

- exp505_exp490_tau500_fade_fixed13_on_exp413 はKaggle private CPU version 1
  （id_no `129519165`）で40 / 40 selector modelsを完走し、technical checksを全PASSした。
  hard OOF `8.243315437`はraw exp501 `8.264890209`から`0.021574771 ft`改善し、4/5 folds、
  固定7 scope、fade top1 pooled `55.2414%` / 5/5 foldsもPASSした。一方、fixed12比tailの
  p95 / worst縮小は`0.000036536 / 0.173168079 ft`で必要な`0.10 / 1.0 ft`を未達。
  `FAIL_CLOSE_WITHOUT_STAGE_D_OR_SAME_OOF_RESCUE`としてStage D、inference、submissionなしで
  終端閉鎖した。

- exp489_acceleration_state_fixed32_mechanism_audit は、ユーザー承認の
  exp458微小数値差waiverを明記し、exp458 v2 fixed4を再利用して残り28 HMM
  wellsをKaggle private CPU version 1（id_no `129171668`）で完走した。
  32 wells / 156,088 rowsのtechnical gateは`10/10 PASS`、decode wall
  `794.097712 sec`、peak RSS `13.971542 GiB`。しかしdirection agreement
  `0.500309` / positive fold `0/5`、forward/persistent episode SSE改善
  `0.4355% / -3.6667%`、persistent改善`8/16 wells` / `2/5 folds`で
  mechanism gateは`2/8 PASS`に留まった。control safetyはPASSしたが
  persistent acceleration機序を支持せず、Stage 1/inference/submissionなしで
  `stage0b_fail_closed`とした。

- exp446_persistent_tvt_rate_exact_hmm はKaggle private CPU version 1
  （id_no `129106260`）で1 candidate × 32 HMM wells、156,088 suffix rowsを
  完走した。constant-Z parity、dense reference、normalization、truth-late、
  SHA readbackはPASSしたが、runtime投影`46,590.855 > 30,600 sec`でtechnical
  `17/18`、mechanismは`0/7`。under-response share削減`-0.061091`、
  forward / persistent SSE削減`-0.306441 / -0.214831`、matched control
  pooled / p95 delta `+7.159063 / +16.310622 ft`だった。known-Z forcingを
  外すpersistent TVT-rate仮説をnegative evidenceとしてfail-closeし、
  Stage 1、rerun、inference、submissionは実行しない。

- exp428_similar_well_gr_registration_map_transfer_readout はcanonical Kaggle private
  CPU version 2（id_no `128932184`）を約`225.6 sec`で完了した。version 1はGR内部
  欠損maskをDTW入力にも適用してsupport 0となる実装差を見つけ、親exp423互換の
  決定的補間へ修正した。version 2はquery support `306 / 773 = 0.395860`で固定下限
  `0.70`をFAILし、評価可能な290 wellsでもrank-1 donor shift MAE `2.529310 ft`は
  zero `1.105172 ft`より`1.424138 ft`悪化、改善`0/5 folds`だった。top-5 oracleも
  zero比`-0.013793 ft`、DTW cost-error Spearman `0.075211`、mean ZNCC gain
  `-0.057438`、local-vs-global block MAE gain `-5.050144 ft`で全て不支持。
  `invalid_or_insufficient_registration_support`として独立rerun、rescue、inference、
  submissionなしでbranchを閉鎖した。

- exp410_likpf_particle_resampling_basin_audit は、exp072 likelihood-PFを
  496 / 496 persistent-offset wellsで500 particles ×128 seedsのまま再生し、
  保存予測との最大差`0.0 ft`で内部stageを監査した。排他的SSE比はfinite
  particle support不足`36.4701%`、across-seed算術平均`36.2441%`、
  within-seed particle平均`10.8561%`、transition`10.7177%`だった。
  hard clamp外の真値、majority-seed resampling extinctionはいずれも0で、
  resampling直後の平均移動も`0.000245 ft`。HMM offsetとはPF SSEの
  `78.9383%`を占める区間が重なり、誤差方向は`90.2655%`一致したが、
  内部mechanism一致は`8.4071%`だけだった。HMMはtransition / priorと
  backward smoothing、PFは有限粒子supportと複数basin平均が主因である。
  追加の固定12 wells ×12 variantsではroughening 10倍がepisode SSEを
  `0.7530倍`、process noise 3倍が`0.8917倍`へ改善した一方、符号検定は
  有意でなくtarget-late。GRほぼ無効は`8.8351倍`、process noise 0は
  `6.2659倍`、resampling無効は`3.4809倍`、clamp 2倍は完全同値だった。
  原因診断として完了し、prediction candidate、inference、submissionはない。

- exp415_fold_safe_rmse_prior_bounded_nudge_on_exp264 は、exp407の悪化を
  candidate一律biasではなくinverse-RMSE task weightによるrow-localな
  score surface / ranking driftと特定した。candidate×fold平均shiftだけなら
  親`8.587004`から`8.580477`へ改善した一方、平均shiftを除いたrow-local成分は
  `8.673599`へ悪化し、final weightとrow-local score差stdのSpearmanは
  `-0.593387`だった。この原因を避け、candidate RMSEをfold-safe additive
  priorとして方向だけに使い、補正を各行`±0.25 ft`へ制限するzero-booster
  policyを固定した。Kaggle private CPU version 1（id_no `128717911`）で
  technical 15/15、scientific 6/6をPASSし、RMSE
  `8.587004 -> 8.563474`（`-0.023530 ft`）、5/5 folds、4/4距離bucket、
  hidden-like 2/2を改善した。785 scopesの数学的risk boundも全PASSし、
  worst-well悪化は`+0.171379 <= +0.25 ft`。保存OOF診断上の方法確立として
  完了し、current-test、route anchor更新、inference、submissionへは進めない。

- exp391_prefix_anchored_mode_persistence_hmm_readout はKaggle private CPU
  version 3（id_no `128527913`）で固定16-well Stage A1を完了した。
  kernel runtime `18105.382 sec`、peak RSS `4.132145 GB`。HMM-supportedは
  `1/19 events = 5.2632%`かつ`1/5 folds`で、事前条件60%・4 foldsをFAIL。
  causeはposterior averaging 1、transition 0、K16 0、fixed blend 3、
  unresolved 15だった。same-pass parity最大`0.350000 ft`、posterior
  normalization誤差`2.4568e-05`、773-well換算`870045.814 sec`もFAILし、
  78,866 candidate rowsは全行saved exp209へfail closedした。threshold /
  tolerance / matching / fallback / blend救済なしでStage B、inference、
  submissionを閉じ、依存するexp395も閉鎖する。

- exp358_exp209_missing_distance_emission_downweight はStage 0の23/23 technical
  checks PASS後、Kaggle private CPU version 2（id_no `128528105`）でStage 1を
  `17475.557881 sec`、1 variant / 5 reporting folds / 773 exact-HMM well-runs、
  parent control再実行0で完了した。candidate RMSE `12.012570`はexp209
  `11.938287`より`0.074283 ft`悪化し、0/5 folds改善、1000+、hidden-like 2面、
  by-well p95/worst、fixed LikPF 50:50を全てFAILした。formal technical gateの
  唯一のfalseはpost-CSV bit-exact weight guardで、753 / 1,200,837 missing
  rowsの最大差は`5.551e-17`だった。科学FAILは独立して明確であり、
  `missing_distance_exp209_failed_close_without_rescue`としてinference、
  submission、rescue、再実行なしで閉じた。

- exp386_cycle_consistent_rgt_scenario_bank はKaggle private CPU version 1
  （id_no `128478384`）の16-well / 5-fold Stage 0を`2411.033 sec`で完了した。
  RGT source coverage `0.989847`、target GR / valid Formation / suffix truth read各0、
  source-valid overlap 0、projected runtime `2867.246 sec`、peak RSS `1.145931 GB`はPASS。
  一方、graph query / scenario-bank / finite-path coverageはすべて0で、cycle residual p95
  `2.363303 > 0.10`もFAILした。全16井戸でscenarioが空のためfull run、Stage 1/2、
  edge/stretch/scenario/diversity救済、inference、submissionなしでFAIL_CLOSEした。
  必須parent bankが存在しないexp387も未実装・未実行で閉じた。

- exp381_formation_contact_order_semimarkov_hmm はKaggle private CPU Stage 0 version 2（id_no `128461656`）を`653.714 sec`、1 diagnostic / 6 surfaces / 5 folds / model・HMM・PF・Beam・booster・control再実行各0で完了した。version 1はsource Formationの列全体欠損を全有限と誤判定し科学処理前に停止、formation別finite outer-train donor固定k=10へ修正した。version 2は773 wells、pre-freeze truth / Formation read 0、期待15 artifactとSHAをPASS。eligible 349/773、1,291 events、crossing MD MAE/p90 `35.994405 / 61.799226 ft`、順序率`0.997135`、constant比`+687.676085 ft`、5/5 foldsはPASSしたが、contact-TVT RMSE `44.770101 ft > 15 ft`だけをFAILした。`stage0_failed_close_without_semimarkov_hmm`として、surface/offset/gate救済、Stage 1、inference、submissionなしでbranchを閉じた。

- exp349_exp287_u_boundary_continuity_fade はKaggle private CPU version 2（id_no `128239658`）を1 fixed postprocess / 5 reporting folds / model・booster・GPU 0で完了した。version 1はpandas返り値型差でfreeze前ERRORとなり、型互換だけを修正した。3,783,989 rows / 773 wells、truth-before-freeze 0を含むtechnical gateは全PASS。exp287 `8.136708220`から`8.135096925`へ`0.001611295 ft`改善し、5/5 folds、0--240 `0.110003778 ft`、hidden-like 2面、far／by-well safetyはPASSしたが、pooled改善下限`0.020 ft`だけをFAILした。1000+は`0.000002 ft`改善で実質無変化だったため、`FAIL_CLOSE_NO_RESCUE`としてcap/tau/threshold/distance/well gate/blend救済、inference、submissionなしでbranchを閉じた。

- exp333_exp226_k16_segment_residual_offset_target は32-well Stage 1 parity/runtime preflightをKaggle private CPU version 1（id_no `128114252`）でPASS後、canonical full train version 1（id_no `128116592`）を1 variant / 1 config / 5 CPU boosters、strict nested exp226 25 fits / 3,865 prediction well-runs、control再学習0で`1,781.997 sec`完走した。CVはexp226 `9.427109597`から`9.076676661`へ`0.350432936 ft`改善し5/5 foldsを改善したが、pooled/near/worstの3 gate FAILでdirect `FAIL_CLOSE_BRANCH`を維持する。その後exp361がfixed12へのadd-one noveltyをPASSした別根拠により、同じexp333内のcurrent-test candidate inferenceをKaggle private CPU canonical version 2（id_no `128368525`）で実行した。exp072正規196列、129 row / 136 model features、exp226 v1 base、保存5 modelを固定SHAで照合し、`65.258 sec`で`14,151 rows / 3 wells / 48 K16 segments`を生成。offsetは`-4.249479～+2.592369 ft`、平均`+0.289689 ft`、candidate decompressed SHAは`7571c628...17cd`。全technical guardをPASSし、新規model/booster・control再学習・selector/blend・submissionは0。version 1の205対196列誤判定は予測前に停止し、version 2でexp072 allowlist適用だけを修正した。候補bankへの組み込み方法は別設計・別承認とする。

- exp322_gr_likelihood_weak_exp226_soft_shrink_readout はKaggle private CPU version 2（id_no `128089589`）を1 candidate / 1 matched control / 5 exp263 strata / 0 model / 0 boosterで195.332秒完走した。version 1は別splitであるexp226元OOF foldとexp263 readout foldの一致を誤要求してscoring前停止し、version 2ではexp263 outer foldをreadout、exp226元foldをsource監査へ分離した。technical hard checksとexp263/cached-exp226 parityは全PASS。一方、発火は4,870行（0.128700%）/10 wellsで事前下限1%/50 wellsをFAILし`INCONCLUSIVE_COVERAGE`。RMSEは`8.238331715 -> 8.239202313`（`+0.000870598 ft`）、activated subsetは`+0.688824530 ft`、改善1/5 folds、1000+ `+0.000966632 ft`、worst well `+0.261431339 ft`、real gainはcircular controlより`-0.001254155 ft`だった。coverageだけでなく科学guardも不支持のため、alpha/quantile/block/clip/emission/selector救済、inference、submissionなしでbranchを閉じた。

- exp312_typewell_group_conditional_gr_emission_table は、exp311 gateをユーザー判断で上書きし、Kaggle private CPU version 1（id_no `128090149`）をscientific 1 + controls 2 / 5 folds / model・booster・decoder 0で完了した。exp293 deployable12固定でglobal-unconditional Student-tと群×GR decile×勾配×欠損のconditional tableを比較したが、MRRは`0.336112 → 0.334519`（`-0.001592`）、top3は`-0.002444`、改善`0/5` folds、group-shuffle差`+0.001611`、hidden-like 2面もFAIL。fallback率`1.823%`とlate-truth境界はPASSし、TVT-shift差は`+0.063809`だったが群条件づけの追加価値は示されなかった。bin/df/kを救済せずbranchを閉じ、exp313〜320の停止を維持する。inference / submissionは未実行。
- exp311_typewell_group_prefix_suffix_gr_calibration_readout はKaggle private CPU version 1（id_no `128085784`）を1 diagnostic / 5 folds / 0 model / 0 booster / 0 decoderで完了した。primary `native_overlap_1` same-group held-out-wellは760/773 wellsで利用可能、identity GR-RMSE `11.745716`から`11.369495`へ`0.376220`改善し、5/5 folds、group-shuffle差`0.240055`、noise R² `0.202320`をPASSした。一方、fit-RMSE R² `-0.003255`とworst-well delta `+12.914716`が固定gateをFAIL。平均的なnoise transferだけではfit品質とtail safetyを保証できずbranchを閉じた。後にユーザー判断でexp312だけを明示上書きして実行したが、exp313〜320は停止を維持する。inference / submissionは未実行。
- exp304_gr_denoiser_emission_separability_readout はKaggle private CPU version 1を完了した。raw MRR/top3
  `0.389626 / 0.452421`に対し、stationary db4 level-3 SWTは`0.424724 / 0.504687`で
  `+0.035098 / +0.052267`改善し、MRR/top3、real-vs-shuffledを5/5 folds、必須4 scope、top1、
  decoy-gapの全gateでPASSした。raw/SWTは全1,546 series technical PASS、RTS/L1はそれぞれ
  1,531 / 974 failuresでtechnical FAIL。`swt_db4_l3`だけを後続のfixed beta 0.15 tempered exact-HMM
  別実験候補へ渡し、RTS variance案は閉じる。HMM/PF/Beam、inference、submissionは未実行。
- exp264_exp263_candidate_confidence_dual_selector の修正版Stage B Kaggle CPU version 5を完了した。
  training-only formation raw/delta 12特徴を除いた88列、12候補、2 objectives、5 foldsの10 boosterで、
  expected-error MAEはprior 5.788783から3.795801、within10 logloss/Brierは0.510131/0.165095から
  0.359972/0.112451へ改善し、全3指標が5/5 foldsで改善してscore guard PASS。hard top1はRMSE
  8.587004でfixed 8.238332より+0.348673、0/5 folds改善、hidden-likeとworst-wellも悪化してFAIL。
  candidate-long 45,407,868行、compact 3,783,989行×74特徴、10/10 model SHAを監査済み。
  confidence groupはpred-abs-error gain 4.267%、`sigma_tvt`は4位・2.958%だった。続く修正版Stage C
  version 6も40 CPU boostersを完走し、expected-error MAE 3.798819、within10 logloss/Brier
  0.359412/0.111830で全指標5/5 folds改善、nested leakageもPASS。40/40 model SHA、25 partition、
  18,919,945 compact rowsを監査した。一方hard top1は8.652532、fixed比+0.414200、改善1/5 foldsでFAIL。
  scoreはhard予測にせず74列compact内部表現に限定した。続くStage D canonical Kaggle T4 version 3は
  clean 273 control 15 + 347列add-only 15 = 30/30 GPU boostersを完走。control 10.476169に対して
  add-only 8.460811、delta -2.015358、5/5 folds、near / mid / 1000+、hidden-like 2面を改善した。
  ただし773 well中255悪化、worst `70925e23` +14.482873で事前上限+0.25を超え、総合guard FAIL。
  compactはadd-only gainの76.9258%、上位4 top1-minus-anchorが61.0343%。corrected inference、
  hard selector、Viterbi、softmax TVT平均、submissionは行わない。
  2026-07-19にcorrected Stage C v6 strict nested scoreとStage D v3 OOFを使うselector-confidence / LikPF
  128-path診断notebookをJupytextから作成し、Kaggle packageをprivate・CPU・internet offで準備した。
  Stage D v3 OOFからviewer用`id,tvt` CSV 3,783,989行 / 773 wellsも生成し、unique ID・finite・viewer loader
  互換とSHA `9fe0cfce...e04b`を確認した。診断notebookは未実行・未pushで、guard判定は変更しない。

- exp275_xgboost_final_regressor_swap_on_exp238 は Kaggle T4 train version 2を完了した。version 1はapproval文字列不一致でデータ読込前に停止しbooster 0本、contractだけを直したversion 2は5/5 XGBoost models、合計2,250 treesを2,984.807秒で完走した。保存済みexp238 415列surface上のpublic Cdeotte v3 XGBoostはRMSE 8.302528478でparent 7.936690031より+0.365838447、5/5 folds、1000+、hidden-like 2面で悪化し、worst wellは+13.880008698。parentとの予測相関0.999995765、固定0.25 blendも7.990746590で+0.054056559悪化した。全raw guard FAILのためtrain-side不採用を維持する。ユーザー承認によるreference inference v2はT4 415.815秒、14,151行、fallback 0、推論時学習0で完了し、rawとparentのtest予測差RMSEは0.917322。raw submission SHA `79452e652e75c3e7f60cb3b77c39dd4f4e175f853f4b1d49accc28b67c70a01c`はsubmit-check FAIL/WARN 0。正規ref `54798185`はPublic LB 7.760で、exp238 hidden-safe 7.775より-0.015改善したが、ML anchor exp274 7.715より+0.045、ensemble anchor exp082 7.601より+0.159悪い。duplicate ref `54798337`も7.760。parameter rescueとanchor更新は行わない。

- exp274_catboost_final_regressor_swap_on_exp238 は Kaggle T4 train version 1を完了した。exp238の保存済み415列surfaceとouter 5 foldsを固定したpublic CatBoost `cb0`はRMSE 8.183503603でparent 7.936690031より+0.246813573悪化し、固定0.25 blendも7.950393906で+0.013703875悪化。改善は1/5 folds、1000+は+0.271066745、hidden-like 2面は約+0.275、worst wellは+12.293691635となり全raw guardがFAILした。ユーザー承認によるreference inference v1はT4 425.779秒、14,151行、fallback 0で完了し、raw CatBoost / parent / fixed0.25 blendの3 submissionはsubmit-check PASS。rawとparentのtest予測差RMSEは1.270216。raw CatBoost submission `ref=54793316`はKaggle API `COMPLETE` / Public LB 7.715で、exp257 7.718を-0.003更新するML submitted anchorとする。一方CVはparentより+0.246814悪化しているためtrain rejectionを維持し、`cb1` / parameter rescueは行わない。後続exp275 XGBoostもnegativeだったため、この保存済みsurface上のCatBoost / XGBoost final-estimator差し替え枝を閉じる。

- exp271_pf_ancc_small_seed_mean_candidate_audit はKaggle CPU train version 2を完了した。version 1はexp072 float32 target復元とexp266 raw TVTの精度差をfail-closedで検出し、raw TVT評価へ直したversion 2は3,783,989 rows / 773 wells、600 particles × 固定8 seeds、runtime 1,386.570秒で完走した。seed0はexp072へ全行差0、mean4/mean8 per-well RMSEはexp266へ最大7.105e-15 ft差、exp263 60 partition SHAもPASS。standalone RMSEはseed0 14.493051、mean4 13.126896、mean8 13.027107。core12へのoracle deltaはrowでmean4 -0.046543 / mean8 -0.049720 / both -0.065252、whole-wellで-0.028392 / -0.036973 / -0.050751。row unique-bestはmean4単独252,772、mean8単独251,635、両方追加時340,687 rowsだった。単一candidateは4 seedへ縮約し、保存済みmean4/mean8とdisagreementを使うadd-only selector監査だけを次候補とする。raw-test PF再生成、inference、submitは行わない。

- exp266_pf_ancc_pf_z_multiseed_stability_audit は Kaggle CPU train v3を完了した。exp072 exact PF ANCC / PF-Zを各600 particles × 64 seedsで3,783,989 rows / 773 wellsへ再生成し、seed 0は両手法とも全行差0、runtime 12,482.144秒。`11d0f5ac`は新規63 seedでもHMM / likelihood-PFへのstrong marginを両手法100%再現し、RMSE 5 ft以下もPF ANCC 98.4%、PF-Z 100%。PF ANCC元seedは新規seed分布のほぼ中央値、PF-Z元seedはむしろ悪い側95.2 percentileであり、単一seed偶然仮説を棄却した。一方、元seedstrong 53 wellsで両手法の過半数seedがstrongなのは21、80%以上再現11、全seed再現4、両手法で80%以上のseedがRMSE 5 ft以下なのは`11d0f5ac` / `fb0904bd`だけだった。raw特徴に明瞭な単一triggerはなく、長いtailはseed誤差/分散を増やすがstrong membershipそのものを説明しない。strong再現率はHMM / likelihood-PFの失敗度とより強く連動し、PF ANCC strong groupにはselection-on-seedも確認した。64 seed meanはPF ANCC 14.493051→12.830319、PF-Z 17.788171→17.074522。直接inference/submitはせず、次候補はPF ANCC固定4/8 seed meanの0-booster candidate監査に限定する。
- exp252_pf_seed_medoid_selectability_audit は Kaggle CPU train v1を完了した。exp243 v3の固定base8 + K8候補とdiagnosticsをPF再実行・学習なしで監査し、3,783,989 rows / 773 wells、5 scope、runtime 86.053秒、全入力SHA guard PASS。K8 candidate内のlikelihood mass / rank / gapは5/5 scopeでshuffled AUCを上回り、whole-well AUC 0.675214 / 0.655102 / 0.654235で順位付け信号を部分支持した。一方、bank最良`resampling_rate`はwhole-well AUC 0.560593、likelihood-mass top1はuseful coverage 0.516043、union-best match 0.280749、best base8比loss平均+3.194947 ftで、base8 fallbackを安全に切れない。3 score単独selectorは不採用だが、base8 fallbackと別bank gateを持つfold-safe二段selectorへのadd-only candidate-ranking特徴量候補にはなり得る。raw PF seed bank + medoid生成はexp243実測37,067.406秒 / 773 wellsで、hidden約200 wellsの単純比例は約2時間40分（未実測）。現時点でraw-test PF再生成、inference、submitは進めない。
- exp231_same_typewell_horizontal_gr_atlas_gated_hmm_emission は Kaggle CPU train v3 を完走した。fold-safe 5 folds / 773 wells / 3,783,989 rowsでsame-typewell horizontal GR atlasをHMM auxiliary emissionへ `alpha=0.025` で加え、gate平均は0.086781で発火した。saved exp072 `likpf_mean` 比のoverall RMSEは 11.594898 -> 11.569950（-0.024947）、MAEは -0.460251、within10は +0.014837だったが、`1000_plus` は 12.702990 -> 12.719560（+0.016570）、316 wellsが悪化し最大well悪化は `b19b0395` の +48.316178 RMSE。persistent-offset onset AUC 0.507654 / q90 lift 1.111111で、hidden-like subgroupも未評価だった。global小改善ではlongtail / worst-well guardを満たさず、exp209 best blend 10.269696も更新しないためtrain-side不採用、raw-test port / inference / submitなしで同atlas emission枝を閉じる。
- exp218_gr_wavelet_rotation_confidence_features_on_exp148 は Kaggle train v1 / inference v1 / submission を完了。exp148 ML anchor に target-free GR wavelet / FFT rotation-denoise confidence features 86列を add-only し、3,783,989 rows / 773 wells / 380 features / 15 boosters で学習した。pooled OOF は `lgb_mean` 8.475793752 で、exp148 GPU `lgb_mean` 8.501281182 から -0.025487430 改善した。inference は current-test GRWR replay と saved `lgb_mean` 15 boosters で 14,151 rows / fallback 0、submission SHA `77a2c2804749dc811ba61f43d9d8827c69282e83e116233559da80b6820c0824`、submit-check pass。submission ref `54457577` は Public LB 7.843 で、exp148 CPU runtime 7.921 から -0.078、exp148 GPU 7.960 から -0.117 改善したため、ML route submitted anchor を exp218 に更新する。overall は exp082 ensemble 7.601 が引き続き最良。
- exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148 は Kaggle CPU train v1 を完了。exp148 ML予測TVTを仮の typewell GR 照合位置にして、3,783,989 rows / 773 wells / 35 feature columns の no-training error detector readout を生成した。exp148 base RMSE は 8.501281182。primary `mlgr_mismatch_signal` の `abs_error_gt10` AUC は 0.573943で採用目安 0.65 に届かず、q90 high-mismatch bucket は error_gt_lift 1.632373 / abs_error_lift 1.425989 と誤差濃縮はあるが単独 detector として弱い。diagnostic correction も base exp148 が最良だったため、add-only LightGBM / inference / submit には進めない。
- exp215_mtp_full_tail_heatmap_path_generator_probe は Kaggle train v1 / T4 GPU を完了。learned `path_logit` を持つ MTP full-tail artifact は rows 18,919,945、unique row ids 3,783,989、row coverage 1.0、fallback unique row rate 0.0 で成立し、exp212 の fallback-heavy / endpoint hold tail 問題を解消した。existing + learned MTP top5 oracle RMSE は 7.434029932 -> 5.113654814、within10 は 0.906525363 -> 0.945863743 と positive。一方、learned MTP top5 only は RMSE 32.333142886、weighted path は RMSE 59.272141581 と弱いため direct replacement / softmax average / PF weight replacement / inference / submit はしない。fallback 0.0 でも生成 path 自体が弱いため、heatmap 由来 path 生成 route は closed/rejected とし、exp204 系 topK candidate / confidence feature follow-up は行わない。
- exp212_heatmap_mdn_full_grid_path_generation_probe は Kaggle train v1 `kentookumura/exp212-hmdn-full-grid-path-generation-train` を完了。exp208 cached dense paths を target-free stitch した sparse source rows から、exp099 feature-cache row grid へ rank1-5 full-grid path artifact を生成した。contract は required columns present、duplicate key rows 0、null required values 0、rows 18,919,945、unique row ids 3,783,989、row coverage 1.0、wells 773 で成立。source coverage は 0.430091631、fallback unique row rate は 0.569908369 で、実 run の fallback は right extrapolated。plot overlay で途中から直線 tail になるのは、exp208 source が `max_tail_rows=2048` までの dense windows に限定され、exp212 がその後ろを endpoint 外挿しているため。existing + stitched top5 oracle RMSE は 7.434029841 -> 5.941479995、within10 は 0.906525363 -> 0.933460166、by-well は 567 improved / 206 same / 0 worse。一方、stitched-only top5 RMSE は 50.085237573 と弱いため direct replacement / softmax average / PF weight replacement / inference / submit はしない。exp215 が full-grid MTP artifact を成立させたが、生成 path 自体の弱さが残ったため heatmap 由来 path 生成 route は closed/rejected とした。
- exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe は Kaggle train v1 `kentookumura/exp208-hmdn-dense-stride-train` を完了。exp202 saved fold model から stride 64 dense local paths を再生成し、25,452 samples / 773 wells / topK10 を作成した。source overlap は exp207 の 3 wells / 39 pairs から 773 wells / 24,679 pairs に増え、row coverage は 0.352337441 -> 0.430091631 に増えた。一方、local topK10 の existing + stitched top3 oracle RMSE は 4.420752853 で exp207 の 4.418699605 を更新せず、stitched only top3 も 47.188322489 と弱い。dense path は物理的に stitch 可能だが direct replacement / softmax average / PF weight replacement / inference / submit には進めない。exp210 で covered-row contract 化、exp212 で full-grid contract 化、exp215 で fallback 0.0 artifact まで完了したが、生成 path 自体が弱いため heatmap 由来 path 生成 route は closed/rejected とした。
- exp203_heatmap_mdn_candidates_into_selector_features は Kaggle train v1 `kentookumura/exp203-hmdn-selector-train` を完了。exp202 heatmap MDN topK を selectable candidate にはせず、既存 exp184 selector の 8 候補に対する add-only `hmdn_` confidence / distance features として追加した。3,783,989 rows / 773 wells / 298 features / 15 boosters、`hmdn_` generated features は 75。best Viterbi は `viterbi_sw050_bias000_jw100_jf025_d0150_std999999_md0000_seg001` で RMSE 10.665741318 / MAE 6.350286735 / within10 0.797977743、path switches 12,807 / 3.384524 per 1000 rows。exp158 continuity 10.789163253 からは -0.123421935 改善したが、exp184 best 10.560650325 からは +0.105090994 悪化し、path switch も exp184 5,713 / 1.509782 より多い。feature-only signal は確認できたが exp184 を更新しないため inference / submit はしない。意図していた heatmap path の candidate 追加は、後続 exp212/215 でも生成 path 自体の弱さが残ったため closed/rejected とした。
- exp202_heatmap_mdn_candidate_generator_probe は Kaggle train v1 COMPLETE。exp182-style 5ch heatmap CNN/MTP を 5 folds / 5 CNN models で学習し、exp099 の既存 PF/Beam candidate union (`pf_ancc`, `beam_mean`, `likpf_mean`, `sc_ens`, `hyb`) に heatmap topK candidates を足す oracle readout を実施した。heatmap only top10 は within10 0.808907780 / oracle RMSE 13.352563025 で existing union 5.068679053 より弱い。一方、existing + heatmap top10 は oracle RMSE 2.745528140 / within10 0.986970985 で、existing union から RMSE -2.323150913、within10 +0.037331362、new-best candidate rate 0.252541120。`1000_plus` bucket は 6.413572416 -> 3.295946470、by-well は 668 improved / 105 same / 0 worse。候補集合としては train-side supported だが、これは oracle headroom なので direct TVT replacement / softmax average / PF weight replacement / submit はしない。後続の exp203 feature-only、exp207/208/210/212/215 path artifact probes まで確認した結果、生成 path 自体が弱いため heatmap 由来 path 生成 route は closed/rejected とした。
- exp201_typewell_spatial_tvt_error_readout は Kaggle train `kentookumura/exp201-typewell-spatial-tvt-error-readout-train-a` v2 を完了。exp148 `lgb_mean` OOF 3,783,989 rows / 773 wells / 54 typewell groups を診断し、baseline RMSE は 8.501281182。same-typewell residual profile corr median は 0.003651、XY 8-nearest bias sign 一致率は 0.494502 で、typewell/XY による形状類似は弱い。一方、offset wells は 66 / 773、top30 high-error wells の 27 wells が offset flag で、well RMSE と abs_bias の相関は 0.948。次は直接補正ではなく、offset / XY outlier / typewell hotspot confidence feature として扱う。
- exp199_typewell_hard_window_pct40_base_surface_keep_exp145_ll_on_exp148 は Kaggle train v1 COMPLETE。exp148 downstream ML surface の base 196 / projection / U-disagreement を exp196 pct40 hard-window cache へ差し替え、exp145 `ll_*` を残す混合 provenance 診断として 3,783,989 rows / 773 wells / 294 features / 15 boosters を学習した。pooled OOF は `lgb0` 8.551067731、`lgb1` 8.533458032、`lgb2` 8.570960612、`lgb_mean` 8.496204218。exp148 GPU `lgb_mean` 8.501281182 から -0.005076964 の小改善だが、`lgb2` は悪化し、改善幅が小さいため直接 inference / submit はしない。ユーザー判断により clean regeneration follow-up もバックログから削除した。
- exp197_cnn_pf_likelihood_probe は Kaggle train v1 COMPLETE。exp099 fixed PF/Beam/likPF candidate cache と raw train local GR/typewell window から candidate-level CNN/SDF likelihood scorer を学習し、point-GR likelihood、exp099 multiobs score、exp111 learned likelihood、likPF baseline、shuffled/no-GR negative control と比較した。real_gr learned_prob candidate AUC は 0.908691639、shuffled_gr 0.902727327、no_gr 0.905303044 で、real-GR の上積みは shuffled に +0.005964、no-GR に +0.003389 と小さい。top1 learned_prob RMSE は 11.301053 で likPF single 11.293248 よりわずかに悪く、exp111 learned probability AUC 0.915825 も下回った。decision は `weak_real_gr_signal_needs_guarded_followup`。PF weight replacement、PF/Beam 再生成、raw-test feature generation、submit へは進めない。
- exp191_typewell_late_range_continuity_selector_on_exp176 は Kaggle train v1 を完了。exp176 v3 saved boosters 15本から OOF score surface を復元し、exp158-style Viterbi 180 variants を評価した。best は `viterbi_sw400_bias000_jw050_jf025_d075_std999999_md0000_seg012` で RMSE 10.598006880 / MAE 6.402336928 / within10 0.793110657。`likpf_mean` 11.594897672 から -0.996890792、exp176 row-wise 10.641296371 から -0.043289491、exp158 best Viterbi 10.789163253 から -0.191156373 改善した。path switches は exp176 row-wise の 261,391 / 69.078 per 1000 rows から 3,620 / 0.957 per 1000 rows へ低下。train-side supported だが near / mid distance bucket は小幅悪化し、by-well 356 wells は exp176 row-wise から悪化したため、selected TVT の direct inference / submit はしない。後続利用は exp148 系 confidence / segment-stability feature surface に限定する。
- exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148 は Kaggle CPU split train v1 を完了。exp148 の `learned_likelihood_confidence` (`ll_*`) を外し、exp191 continuity selector の predicted-error surface、selected candidate family、typewell pct context、segment stability / boundary risk features に置き換える replacement-only 実験。CPU timeout 対策として Kaggle train notebook は `train_lgb0` / `train_lgb1` / `train_lgb2` に分割し、各 1 config x 5 folds、合計 15 boosters、parent/control 再学習なし。pooled OOF は `lgb0` 9.464292702、`lgb1` 9.331742862、`lgb2` 9.313152706、3 split 平均 `lgb_mean_split3` 9.321908826。exp148 `lgb_mean` 8.501281182 から +0.820627644、exp193 `lgb_mean` 8.456665439 から +0.865243388 悪化したため、replacement-only 仮説は完了/不採用として backlog から外し、current-test feature generation / inference port / submit はしない。
- exp195_denoised_calibrated_matching_replacement_only_on_exp148 は Kaggle train v1 を完了。exp190 add-only の `lgb1` 改善を受け、exp148 の `learned_likelihood_confidence` (`ll_*` 54列) を外して `denoised_calibrated_matching` block に置き換える replacement-only 実験を評価したが、pooled OOF は `lgb0` 9.612543035、`lgb1` 9.405030561、`lgb2` 9.388749748、`lgb_mean` 9.409612611。exp148 `lgb_mean` 8.501281182 から +0.908331429、exp190 add-only `lgb_mean` 8.503596159 から +0.906016451 悪化したため train-side rejected とし、current-test feature generation / inference port / submit は行わない。
- exp190_denoised_calibrated_matching_features_on_exp148 は Kaggle train v1 を完了。exp148 learned-likelihood ML anchor に raw / rolling median / Savitzky-Golay の GR shift-scan sharpness、posterior ambiguity、candidate disagreement、prefix backtest quality を add-only で追加し、3,783,989 rows / 773 wells / 431 features / 15 boosters を学習した。pooled OOF は `lgb0` 8.601678275、`lgb1` 8.539624480、`lgb2` 8.540073562、`lgb_mean` 8.503596159。`lgb1` 単体は exp148 同 config から -0.024346641 改善したが、採用基準の `lgb_mean` は exp148 `lgb_mean` 8.501281182 から +0.002314978 悪化した。したがって train-side rejected とし、current-test parity 実装、inference port、submit はしない。
- exp186_typewell_late_range_pfbeam_generation_soft_prior は Kaggle train v3 を完了。v1/v2 の 192-row prefix-holdout audit は意図した full replay cache rebuild ではなかったため superseded とし、v3 で raw train horizontal/typewell から exp072-style full replay train feature cache を作り直した。既存 full replay cache は generation input として使っていない。出力は 3,783,989 rows / 773 wells / 196 features、runtime は summary 15,783.764 sec、feature generation 14,053.477 sec。selected soft prior は `pct50_strong2_pct70_weak0p5` で、PF_ANCC、PF_Z、Beam、128-seed likelihood-PF に適用した。train feature raw gzip SHA は `4bb7a43278ec65143d61c3451353735093995d5258aad665b901237a6a469185`、decompressed SHA は `b4dd75312d91b21f55b8d1ad09a8590c6bb75857ddfbbbc84d7db175dbb75d15`。exp072 full replay cache との direct PF/Beam RMSE TVT 比較では `pf_ancc` が 14.493061 -> 14.220030、`beam_mean` が 15.774328 -> 15.753703 と小改善した一方、最強候補 `likpf_mean` は 11.594898 -> 12.942278 へ +1.347381 悪化した。したがって exp072 replacement としては不採用で、model training / inference / submit には進めない。
- exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector は Kaggle train `kentookumura/exp183-copcf-train` version 2 を完了。v1 は fold0 multiclass 後の candidate-long feature 生成中に `DeadKernelError` で落ちたため、v2 は long-model memory guard に寄せ、train/eval long sample を 120k rows/fold、full valid OOF を 50k row chunk prediction に変更した。best Viterbi は RMSE 10.601481774 / MAE 6.386571251 / within10 0.792418794、exp158 continuity 10.789163253 から -0.187681479 改善。path switches は 5,650 / 1.493 per 1000 rows。train-side supported for review だが、inference port / submit 前に raw-test parity、worst-well / bucket / exp115 subgroup、必要なら高メモリまたは split train を同じ exp183 内で確認する。
- exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158 は Kaggle train `kentookumura/exp184-hmpf-train` v2 を完了。v1 は fold0 multiclass 後に `DeadKernelError` だったが、v2 で exp183 と同じ long-model memory guard に寄せ、train/eval long sample 120k rows/fold と 50k row chunk prediction で完走した。best Viterbi は RMSE 10.560650325 / MAE 6.329187986 / within10 0.797056492、exp158 continuity RMSE 10.789163253 から -0.228512928 改善。path switches は 5,713 / 1.509782 per 1000 rows。heatmap sparse distance q4 は RMSE 14.058409、exp115 spatial valid は 12.696140、typewell purged valid は 12.629861 と stress が残るため、train-side supported だが inference port / submit 前に raw-test heatmap parity、sparse interpolation coverage、hidden-like subgroup guard を同じ exp184 内で確認する。
- exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe は Kaggle train v1 / T4 GPU で完了。24 CNN models、773 usable wells。full-fold `base_real_w128_b64_fullfold` は top3 within10 0.500000、top10 0.808908、top10 oracle RMSE 13.296284 で、`base_shuffled_w128_b64_fullfold` 0.218536 と `base_no_gr_w128_b64_fullfold` 0.071429 を大きく上回った。GR signal は full-fold でも支持される。一方、`geometry_real_w128_b64_fullfold` は top3 0.487710 で base より -0.012290、`geometry_real_w256_b96_fold01` も 0.417512 と弱く、geometry / larger window は採用しない。worst-well top3 0.0 が残るため、full-length inference、direct replacement、submit には進めず、heatmap path feature / selector 材料に限定する。
- exp179_cnn_sdf_mtp_heatmap_probe は Kaggle train v2 / T4 GPU で完了。v1 は P100 割当で PyTorch 2.10 が `sm_60` 非対応だったため失敗し、v2 で `machine_shape=NvidiaTeslaT4` を明示した。5ch heatmap CNN/SDF/MTP smoke は train 2,304 samples / valid 512 samples / 32 valid wells、target-in-grid rate 1.0。`real_gr` は top3 within10 0.44921875、top10 within10 0.794921875、top10 oracle RMSE 14.071006。`shuffled_gr` は top3 0.232421875、top10 0.541015625、`no_gr` は top3/top10 0.0625 だった。real GR は shuffled-GR を top3 +0.216796875、no-GR を +0.38671875 上回り、GR signal を使えている smoke と判断する。ただし 1 fold / selected wells / fixed 128x64 window の診断なので、direct replacement、inference port、submit はしない。
- exp175_cluster_outlier_typewell_prior_gate は Kaggle train v2 を完了。exp148 `lgb_mean` baseline は RMSE 8.501281182 / MAE 5.335650953 / within10 0.856332035 で、best policy は補正なし baseline のままだった。best non-baseline は `typewell_native_overlap_0p999__own_z_gt2p0__std_le20__a0p05__c5` だが RMSE 8.501592821、baseline から +0.000311639 悪化した。by-well regression は最大 +0.181173 RMSE に抑えられ、near `000_050` bucket は -0.004189 改善したが、global / exp115 hidden-like stress では支持されない。したがって exp148/exp092 ML output への cluster-outlier gated prior posthoc correction は完了/不採用とし、inference port / submit はしない。PF/Beam/likPF 候補対象の未検証点は exp181 で確認済み。
- exp181_cluster_outlier_pfbeam_prior_gate は Kaggle train v1 を完了。`likpf_mean` baseline は RMSE 11.594897672 / MAE 7.067632584 / within10 0.772807479。best gated `any_outlier_signal_k8/std_le20/a0.2/c40` は RMSE 11.479140438、delta -0.115757234、within10 0.775467371、gate rows 908,309 / wells 215。exp109 global reference の max well regression +6.594183 は +4.359666 まで下がり、clip20 では +3.032388 まで下がるが、direct posthoc correction としてはまだ大きい。distance bucket と exp115 hidden-like stress は壊れていないため prior signal は有効だが、inference port / submit はしない。`cluster_outlier_pfbeam_prior_gate` backlog は完了として外す。
- exp174_typewell_late_range_ml_posthoc_clip_audit は Kaggle train v1 を完了。exp148 `lgb_mean` OOF 3,783,989 rows / 773 wells に対し、`known_last_pct >= 0.75/0.80` かつ `pred_pct < lower_bound` の row だけ shrink / clip する no-training grid を監査した。baseline は RMSE 8.501281182。lower bound `0.55/0.60/0.65` は changed_rows 0 で no-op。発火する best policy `fixed_lb0p7_klp0p75_a0p25` は changed_rows 2,098 / 2 wells、RMSE 8.501891 で +0.000609 悪化し、最大発火 `known_last_m0p05_klp0p75_a0p25` は changed_rows 13,657 / 14 wells、RMSE 8.518425 で +0.017144 悪化した。したがって `typewell_late_range_ml_posthoc_clip_audit` は完了/不採用として backlog から外し、inference port / submit はしない。typewell late-range prior を続ける場合も hard ML posthoc ではなく、PF/Beam candidate feature / selector prior に限定する。
- exp160_sp45_bimodal_selector_confidence_features_on_exp148 は Kaggle train v2 / inference v1 / scoring を完了。v1 train は SP45 feature generation 中の DataFrame fragmentation / kernel death で失敗したため、v2 で dict-of-array 生成と row-order concat に修正した。train は 3,783,989 rows / 773 wells / 372 features / 15 boosters、feature join coverage pass。pooled OOF は `lgb0` 8.582750400、`lgb1` 8.458535254、`lgb2` 8.502983731、`lgb_mean` 8.463718774。exp148 `lgb_mean` 8.501281182 から -0.037562408 改善したため train-side positive。inference v1 は 14,151 rows、fallback 0、submission sha256 `366543ab052b98afec8c61f020c6eccc84c751fd734262dd9913bbb53fab354b`、submit-check pass。ユーザー確認により exp160 の Public LB は ref `54183128` の 8.061。exp148 Public LB 7.960 から +0.101 悪化したため採用しない。
- exp128_trajectory_local_typewell_self_gr_switch_audit は Kaggle train v2 を完了。v1 は soft blend の `0 * NaN` 伝播で coverage 0.756345750 の subset 評価になったため invalid とし、v2 を正式結果とする。v2 は 3,783,989 rows / 773 wells で best が baseline `likpf_mean` RMSE 11.594897672、delta 0.0。`finite_self_prior_rate` は 0.756345750 だが、平均 `typewell_cost - self_cost` は -0.742965639 で self-GR cost が typewell cost より悪く、switch / blend gate は 0.0 のまま発火しなかった。`self_gr_prefix_prior_tvt` は worst-well で数千 ft 規模に壊れるため、trajectory local switch / direct self-GR path / inference port / submit はしない。
- exp126_exp073_exp092_pf_beam_pseudotail_failure_map は Kaggle train v2 を完了。3,783,989 rows / 773 wells の同一 pseudo-tail surface で、exp092 `lgb1` は exp073 `lgb_mean` 9.526375 から 9.322480 へ -0.203895 改善した。一方、near prefix は悪化し、`distance_bucket=000_050` は +0.347196、`000_050 + pf_dense_diff_q4` は +4.183256。`1000_plus + pf_dense_diff_q4` は -0.492495 で、exp092 の改善は longtail / high disagreement に寄る。oracle candidate RMSE は 5.872226 と大きい headroom があるが、target 依存のため direct gate / submit には使わず、confidence feature / segment-level selector 診断に限定する。
- exp135_tvt_dense_high_drift_confidence_gate_on_exp092 は Kaggle train v2 を完了。exp092 base RMSE 9.322480 が最良で、best gate `seg_dense50_q75_tail1000_min100_clip20_a050` は RMSE 9.874846、+0.552366 悪化した。PF `likpf_mean` worst50 や common PF+ML worst26 では dense 候補の救済 headroom があるが、configured gate は global OOF、near rows、1000+ bucket、worst-well regression を壊すため direct gate / inference port / submit はしない。
- exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit は Kaggle train v2 を完了。既存 exp118 との番号衝突は local canonical を exp119 に改番して解消し、旧 Kaggle kernel id `kentookumura/exp118-same-typewell-prefix-gr-transfer-train` の履歴として v1/v2 を保持する。v2 best は baseline `likpf_mean` RMSE 11.594897672 で、same-typewell GR match best は 11.614959308 と悪化したため、direct correction / candidate path / inference port / submit はしない。
- exp120_typewell_geology_neighbor_prior_pf_likelihood_probe は Kaggle train v3 を完了。best `neighbor_drift_prior` は `likpf_mean` RMSE 11.594897884 から 11.207143527 へ -0.387754357 改善し、`1000_plus` longtail でも 12.704015404 -> 12.280217665 と -0.423797740 改善した。一方で marker likelihood は悪化し、by-well 最大悪化 +6.323216 が残るため direct correction / inference port / submit はしない。
- exp001_baseline の `last_anchor` は full CV OOF RMSE 15.909853。公開 notebook 調査の 15.91 と一致し、評価 mask と baseline 実装の sanity check になった。
- 同じ exp001 の `recent_linear` は OOF RMSE 41.022355 で悪化。単純 slope 外挿ではなく、`last_anchor` からの drift / residual を学習する方針を優先する。
- exp002_drift_minimal は high priority の residual 学習を実装し、Kaggle full CV で `drift_hgb` 14.124569、public LB 12.533。exp001 の CV 15.909853 / public LB 15.883 から大きく改善した。
- exp003_residual_ablation は sampling / shrink / feature ablation を実行し、`feature_no_gr_signal` が CV 13.882944 で最良。exp002 から CV は 0.241625 改善したが、public LB は 12.852 で exp002 の 12.533 より悪化した。
- exp004_gr_gating は GR coverage が弱い well だけ no-GR に切り替える `gate_low_gr_coverage_hard` が gating variants で最良、CV 13.932968。exp002 から 0.191601 改善したが、全体最良は exp003 相当の `control_exp003_no_gr` 13.882944。
- exp004 の inference / submit-check は完了。visible aggregate RMSE は exp004 7.948310、exp002 7.916353、exp003 8.472623 で、exp004 は exp003 より改善したが exp002 よりわずかに悪い。
- exp004_gr_gating の提出は public LB 12.730。exp003 12.852 より改善したが exp002 12.533 には届かず、LB anchor は引き続き exp002。
- exp005_gr_gate_recalibration は selected `gate_low_gr_strict_hard` が CV 13.936732 / Public LB 12.579。exp004 selected gate 再現の CV 13.932968 より 0.003764 悪いが、visible `000d7d20` を no-GR routing から外し、Public LB は exp004 12.730 から改善した。ただし exp002 12.533 には届かず、LB anchor は exp002 のまま。
- exp006_hard_well_router_diagnostic は Kaggle full train を完了し、selected `gate_low_gr_strict_hard` は CV 13.936732。hard/no-GR 候補 248 wells、public-like/all-GR 維持候補 193 wells をタグ付けした。best inference-safe rule は exp004 相当の `low_gr_any_to_no_gr` CV 13.932968、oracle best-of all-GR/no-GR は 13.299351 で router headroom が残る。
- exp007_hard_well_router は selected `hard_router_low_gr_guarded` が CV 13.921559 / Public LB 12.675。exp005 guarded から CV は 0.015173 改善し、route counts は all-GR 476 wells / no-GR 247 wells / guarded 50 wells。ただし Public LB は exp005 12.579 と exp002 12.533 に届かず、LB anchor は exp002 のまま。
- exp008_gr_ncc_matcher は typewell / horizontal GR の multi-scale NCC を add-only 特徴として検証し、selected `gr_ncc_no_gr_multi` は CV 14.641514。best control の `control_exp003_no_gr` 13.882944、raw-GR control の `control_exp002_all` 14.124569 のどちらにも届かず、提出しない。
- exp009_formation_surface_guide は train-only formation columns を fold-safe KNN surface guide に変換して検証したが、selected `formation_knn_no_gr` は CV 14.558630、`formation_knn_all` は 14.739226 でどちらも control より悪化。提出しない。
- exp010_trajectory_drift_ablation は selected `trajectory_full_no_gr` が CV 14.236694 で悪化。best は `control_exp003_no_gr` 13.882944、trajectory variants はすべて exp003 no-GR control より悪く、提出しない。
- trajectory_feature_error_audit では exp010 の well metrics を診断し、full trajectory は hard-no-GR 候補 248 wells で 16.721940 -> 17.688955、steep trajectory 186 wells で 15.208859 -> 15.979471、high GR missing 293 wells で 13.250584 -> 13.959989 と悪化が強い。一方 `public_like_keep_all_gr` 193 wells では 14.684601 -> 14.289867 と改善したため、trajectory signal は add-only ではなく selector / router 用に限定する。
- exp012_single_catboost_lightgbm_residual は HGB / LightGBM / CatBoost の all-GR/no-GR 単体 model-class ablation を Kaggle full CV で実行し、`lightgbm_no_gr` が CV 13.549257 で best。`control_hgb_no_gr` 13.882944 から 0.333687 改善し、Public LB 12.320 で従来 anchor `exp002` 12.533 も更新した。
- exp013_model_diversity_or_postprocess は Kaggle full CV / inference / submit-check / submit を完了。raw `lightgbm_no_gr` は clean CV 13.549257 を再現し、`distance_bucket_shrink_fit` は OOF-fit score 13.501824。Public LB 12.271 で exp012 12.320 を更新した。後処理 alpha は同じ OOF rows で fit/evaluate しているため、same-OOF score として分けて扱う。
- exp014_postprocess_cv_audit は `exp013` の row OOF artifact を監査し、bucket alpha の leave-one-original-fold-out CV 13.535596、well-bucket holdout 13.510690 を確認した。raw 13.549257 から fold 外でも改善は残るが、same-OOF 13.501824 は楽観値として扱う。
- exp015_public_pf_beam_scale_selector_features は `control_lightgbm_no_gr` 13.549257 を再現したが、`pf_beam_no_gr` は CV 14.442743 で大きく悪化。PF/beam add-only features は採用せず、再検討する場合は candidate quality audit / feature pruning / router に限定する。
- exp016_public_postprocess_ablation は exp013 OOF 上で public-style 後処理を切り分けた。same-OOF では `exp013_bucket_shrink` 13.501824、固定候補では `alpha_tau_250_a020_115` 13.515133 が改善したが、leave-one-original-fold-out candidate selection は 13.551561 で raw 13.549257 より悪かった。
- exp018_candidate_distribution_router は既存 OOF 候補 router を監査し、same-OOF best `disagreement_damped_raw` は 13.537122 で raw 13.549257 を上回ったが、leave-one-original-fold-out selection 13.644470、well-hash holdout selection 13.646503 で悪化。candidate router は提出実装へ進めない。
- exp019_pf_beam_candidate_quality_audit は Kaggle full audit で 773 wells / 3,783,989 rows を診断し、raw `lightgbm_no_gr` 13.549257 が best。best PF-derived full-row candidate `pf_hold_mean_blend` は 19.142388、`pf_best` は 114.654448、`exp015` PF feature model mean well delta は +0.648761 で、PF/beam direct candidates / features / router 再投入はしない。
- exp020_distance_weighted_training_audit は Kaggle full CV で `control_lightgbm_no_gr` 13.549257 を再現し、best `near_down_far_up_lightgbm` が 13.470015。raw から -0.079242、exp014 held-out postprocess 13.535596 から -0.065581 改善した。near rows は悪化したが、rows 1000-2499 と 2500+ の改善で全体を改善したため、次は inference / postprocess / submit 候補化を優先する。
- exp021_distance_weighted_inference_postprocess は Kaggle train / inference / submit を完了。`weighted_raw` は exp020 と同じ 13.470015、`weighted_distance_bucket_shrink` は 13.415799 で best。rows 0-49 は 3.576164 -> 0.963697、rows 50-249 は 4.078820 -> 3.572572 と near-row 悪化を補正したが、Public LB は 12.523 で exp013 12.271 より悪化。Public LB 基準は更新しない。
- exp025_pseudo_tail_postprocess_cv_audit は Kaggle full CV で raw pseudo-tail 12.942938 を再現し、fixed `exp014_bucket_shrink_params` が 12.870780 で best held-out candidate。fixed-candidate selection は original-fold / well-hash の全 holdout で同 candidate を選び、raw から -0.072158 改善した。same-OOF bucket alpha fit 12.863570 は診断値として分け、次は fixed bucket shrink の inference / submit 化を優先する。
- exp026_pseudo_tail_bucket_shrink_inference_submit は exp025 selected fixed bucket shrink を Kaggle inference / submit 化し、Public LB 12.102 (`ref=53411137`) で exp024 raw 12.166 から -0.064 改善した。14,151-row `submission.csv` は submit-check PASS。prediction range は 11590.725143 - 12237.368348、exp024 raw submission との差分 RMSE は 0.611885。
- exp050_xgboost_pseudo_tail_inference_submit は exp049 XGBoost pseudo-tail fixed bucket-shrink を Kaggle inference / submit 化し、Public LB 12.083 (`ref=53521999`) で exp026 12.102 から -0.019 改善した。14,151-row `submission.csv` は submit-check PASS。prediction range は 11587.960181 - 12234.905349、exp026 との差分 RMSE は 1.431860。
- exp051_pseudo_tail_lgbm_param_micro_tune は Kaggle train version 1 を完了し、`lgbm_capacity_leaves47_minchild60_exp014_bucket_shrink_params` が CV 12.634392 で best。same-run control fixed bucket-shrink 12.784540 から -0.150148、exp049 XGBoost fixed bucket-shrink 12.779452 から -0.145060 改善した。Public LB は未確認のため、次は選択候補だけを inference port して submit-check と予測差分を確認する。
- exp027_public_replay_needless090_sel15_spread3 は公開 `needless090/lb8-781-rogii-sel15-spread3` を replay し、Kaggle inference version 1 と submit-check が完了。UI submit ref `53420592` は Public LB 8.781 で、source title score を再現し、現時点の Public LB 基準を更新した。regular file-upload CLI submit は Notebook-only code competition では API 400 になるため、今後は `task submit-code` の `-k/-v/-f submission.csv` 形式を使う。
- exp028_public_replay_second_sel15_or_blend_audit は公開 `needless090/lb-8-860-rogii-sel15-256seeds` を replay し、Kaggle inference version 2 と submit-check が完了。出力 SHA256 は exp027 と同一、差分 RMSE 0.000000 / corr 1.000000 だったため、新規提出と public replay blend は行わない。
- exp030_public_sel15_pf_candidate_selector は exp029 の 1,782,279-row OOF-like artifact を監査し、fixed `pf090_hold010` が same-OOF 15.089532、original-fold selection 15.141132、well-hash selection 15.131490 で raw public PF 15.172636 を上回った。一方、信頼度による代替処理と bucket ごとの固定選択は不安定で、well-hash bucket selection は 15.183372 と悪化したため、次は hard selector ではなく固定 90% PF / 10% hold blend の inference 差分監査に進む。
- exp031_public_sel15_pf_hold_blend_inference_audit は exp027 公開 sel15 inference flow に fixed `pf090_hold010` を実装し、Kaggle kernel version 1 と submit-check が完了。public sample は全 wells が visible train wells で 変更なしの物理モデル処理を使うため exp027 と完全一致し、SHA256 `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`、差分 RMSE 0.000000。見えない test well 用処理の code-submit ref `53443300` は Public LB 8.956 で exp027 8.781 から +0.175 悪化したため、fixed `pf090_hold010` 見えない test well 用処理は採用しない。
- exp032_public_sel15_pf_residual_correction は exp029 の 1,782,279-row OOF-like artifact で `target_tvt - pf_pred` residual を学習し、selected `ridge_residual_shrink0p5_clip20p0` が original-fold OOF 14.937393、well-hash holdout 14.844228。raw public PF 15.172636 と fixed `pf090_hold010` 15.089532 を両 holdout で上回り、全距離 bucket でも public PF から改善した。ただし original-fold split 3 は raw PF より +0.074417 悪化したため、次は selected candidate の inference 移植と差分監査に限定する。
- exp033_public_sel15_pf_residual_inference_port は exp032 selected `ridge_residual_shrink0p5_clip20p0` を public sel15 inference flow に移植し、Kaggle inference version 1 と submit-check が完了。exp029 artifact 1,782,279 rows から 473,950 rows を sampling して Ridge residual model を fit した。public sample は全 wells が visible train wells で 変更なしの物理モデル処理を使うため exp027/exp031 と完全一致し、SHA256 `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`、changed_rows 0、diff RMSE 0.000000。code submit ref `53444678` は Public LB 14.961 で exp027 8.781 から +6.180 悪化したため、residual correction 見えない test well 用処理は採用しない。
- 2026-05-27 に ROGII 公式情報を取得し、評価指標は RMSE、提出形式は `id,tvt`、提出は Notebook-only と確認した。
- 検証は `well_id` の GroupKFold を基本にし、`TVT_input` が NaN の evaluation zone だけを score する。
- train-only formation columns は隠しテストで使えない前提のため、初期ベースラインでは直接使わない。

## 変更履歴

- 2026-06-16: exp068_equivalent_pixiux_inference_port はユーザー指示により破棄。CV-only train v4 と invalid ref `53654439` は履歴として残し、修正版 Kaggle rerun は行わない。代替として `backlog/KAGGLE_DIRECTION.md` に backlog `exp073_exp039_cv_reassessment` を追加し、元バックログの対象だけを exp063 から exp073 に差し替えた。
- 2026-06-14: exp068_equivalent_pixiux_inference_port をレビュー結果に基づき修正。目的を exp068 再学習モデルの提出に固定し、train notebook に full LightGBM booster 保存を追加、inference notebook を exp068 full model artifact + hidden-test exp063 replay feature generation に変更した。旧 inference v2 / ref `53654439` の Public LB 762.715 は静的 exp063 public-sample prediction artifact 依存による hidden fallback 疑いとして採用しない。この設計変更は 2026-06-16 に exp068 ごと破棄済み。
- 2026-06-14: exp068_equivalent_pixiux_inference_port の Kaggle train v4 を完了。exp039/exp038 系 CV surface に exp063 tracker/PF/Beam output features を join し、exp063 Pixiux LightGBM family を再学習評価した。`lgb_mean` は original-fold 11.878856、well-hash 11.994729。joined rows は 1,781,963、features は 65。この v4 は full model artifact 保存前の CV-only 実行。
- 2026-06-14: exp068_equivalent_pixiux_inference_port の旧 Kaggle inference v2 と submit-check を完了。`submission.csv` は 14,151 rows、fallback 0、SHA256 `26e3238a29ff37d4193cfec073d507fc840082b33fd82be10a0cc619302739c4`。public sample 上では exp063 inference v2 submission との差分が RMSE 0.000277 / max abs 0.000484 で丸め差程度に同一だったが、code-submission hidden scoring 用としては不適切なため current flow から廃止。
- 2026-06-07: exp033_public_sel15_pf_residual_inference_port の Kaggle inference version 1 と code-submit ref `53444678` を完了。`submission.csv` は 14,151 rows、submit-check PASS、prediction range 11587.038593 - 12240.016066、SHA256 `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`。public sample は exp027/exp031 と完全一致、監査 summary は changed_rows 0 / changed_wells 0 / diff RMSE 0.000000。Public LB は 14.961 で exp027 8.781 より大きく悪化。
- 2026-06-07: exp032_public_sel15_pf_residual_correction の Kaggle train version 2 を完了。`ridge_residual_shrink0p5_clip20p0` が original-fold OOF 14.937393、well-hash holdout 14.844228 で selected。artifacts と `metrics.json` を `/tmp/kaggle-output/exp032_public_sel15_pf_residual_correction/train_v2` から同期し、`public_sel15_pf_residual_correction` backlog を実装済みとして削除、次候補 `public_sel15_pf_residual_inference_port` を追加。
- 2026-06-07: exp031_public_sel15_pf_hold_blend_inference_audit の Kaggle inference version 1 と code-submit ref `53443300` を完了。`submission.csv` は 14,151 rows、submit-check PASS、SHA256 `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`。public sample は exp027 と完全一致、監査 summary は changed_rows 0 / changed_wells 0 / diff RMSE 0.000000。Public LB は 8.956 で exp027 8.781 より悪化。
- 2026-06-07: exp030_public_sel15_pf_candidate_selector を実行。raw public PF selector は 15.172636、best same-OOF は `pf090_hold010` 15.089532、original-fold candidate selection 15.141132、well-hash candidate selection 15.131490。bucket selection は well-hash 15.183372 で raw より悪化したため、hard selector は不採用。
- 2026-06-07: exp027_public_replay_needless090_sel15_spread3 の UI submit ref `53420592` が COMPLETE。Public LB 8.781 で source title score を再現し、現 Public LB 基準を更新。`submission.csv` は 14,151 rows、submit-check PASS、SHA256 `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`。
- 2026-06-07: exp028_public_replay_second_sel15_or_blend_audit の Kaggle inference version 2 を完了。`submission.csv` は 14,151 rows、submit-check PASS、SHA256 `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`。exp027 と完全一致したため提出なし。
- 2026-06-06: exp026_pseudo_tail_bucket_shrink_inference_submit の Kaggle inference / submit を完了。postprocess `distance_bucket_shrink`、submission 14,151 rows、submit-check PASS。提出 ref `53411137`、Public LB 12.102。prediction range 11590.725143 - 12237.368348、exp024 raw との差分 RMSE 0.611885。
- 2026-06-06: exp025_pseudo_tail_postprocess_cv_audit の Kaggle full CV を完了。raw pseudo-tail は 12.942938、fixed `exp014_bucket_shrink_params` は 12.870780、original-fold bucket alpha fit 12.887830、well-hash bucket alpha fit 12.879401、same-OOF bucket alpha fit 12.863570。提出なし、次は selected fixed bucket shrink の inference / submit 化。
- 2026-06-06: exp021_distance_weighted_inference_postprocess の Kaggle train / inference / submit を完了。`near_down_far_up_lightgbm` weighted raw は 13.470015、`weighted_distance_bucket_shrink` は 13.415799 で exp020 から -0.054216 改善。inference submission は 14,151 rows、submit-check PASS。提出 ref `53406803`、Public LB 12.523 で exp013 12.271 より悪化。
- 2026-06-06: exp020_distance_weighted_training_audit の Kaggle full CV を完了。`control_lightgbm_no_gr` 13.549257 を再現し、`near_down_far_up_lightgbm` が 13.470015 で best。`far_upweight_lightgbm` 13.550841、`near_downweight_lightgbm` 13.580536、`near_mid_far_segmented_lightgbm` 13.655239。小さい artifact と log を保存。提出なし、次は selected weighted model の inference/postprocess 化。
- 2026-06-06: exp019_pf_beam_candidate_quality_audit の Kaggle full audit を完了。PF/beam direct candidates、scale、confidence、GR missing、eval length、Z span、trajectory 条件を `exp013` raw OOF と比較し、raw `lightgbm_no_gr` 13.549257 が best。PF-derived full-row best は `pf_hold_mean_blend` 19.142388、`pf_best` 114.654448、PF feature model mean well delta +0.648761。提出なし、PF/beam 再投入なし。
- 2026-06-05: exp018_candidate_distribution_router を実行。`exp013` raw LightGBM、HGB control、`exp017` DTW/DWT OOF を結合し、fixed/blend/distance/disagreement router を比較。same-OOF best は `disagreement_damped_raw` 13.537122 だが、original fold 外 selection 13.644470、well-hash holdout 13.646503 で raw 13.549257 より悪化。PF/beam row OOF はローカルにないため optional skip。提出なし。
- 2026-06-05: exp016_public_postprocess_ablation を実行。`exp013` OOF の `lightgbm_no_gr` 3,783,989 rows で SG smoothing / fade-in / hold blend / alpha-tau / bucket shrink を比較し、same-OOF best は `exp013_bucket_shrink` 13.501824、固定候補 best は `alpha_tau_250_a020_115` 13.515133。original fold 外 selection は 13.551561 で raw 13.549257 に届かず、提出なし。
- 2026-06-05: exp017_deterministic_dtw_addonly の Kaggle full CV を確認。`control_lightgbm_no_gr` は CV 13.549257 を再現、`dtw_dwt_no_gr` は 13.949718、bucket postprocess 後も 13.910963 で悪化。提出なし。1.1GB の `row_oof_predictions.csv` は `/tmp/kaggle-output/exp017_deterministic_dtw_addonly/train/artifacts/` に保持。
- 2026-06-04: exp015_public_pf_beam_scale_selector_features の Kaggle full CV を完了。`control_lightgbm_no_gr` は 13.549257 を再現、`pf_beam_no_gr` は 14.442743 で悪化。提出なし。小さい artifact と log を保存。1.1GB の `row_oof_predictions.csv` は後続入力として使っていないため 2026-06-05 に `/tmp` から削除し、必要時は Kaggle output から再取得する。
- 2026-06-04: exp014_postprocess_cv_audit を実行。`exp013` の `row_oof_predictions.csv` から `lightgbm_no_gr` を監査し、raw 13.549257、same-OOF bucket fit 13.501824、leave-one-original-fold-out 13.535596、well-bucket holdout 13.510690 を記録。提出なし。
- 2026-06-04: exp013_model_diversity_or_postprocess の Kaggle full CV / inference / submit-check / submit を完了。raw `lightgbm_no_gr` は clean CV 13.549257、`distance_bucket_shrink_fit` は OOF-fit score 13.501824 / Public LB 12.271。小さい artifact と log を保存し、1.1GB の `row_oof_predictions.csv` は `data/external/kaggle-output/exp013_model_diversity_or_postprocess/train/artifacts/` に保存。
- 2026-06-03: trajectory_feature_error_audit を実行し、`experiments/exp010_trajectory_drift_ablation/artifacts/trajectory_feature_error_audit/` に well deltas、group summary、top hurt/help、metrics、report を保存。
- 2026-06-02: exp010_trajectory_drift_ablation の Kaggle full CV を完了。selected `trajectory_full_no_gr` は CV 14.236694、best は `control_exp003_no_gr` 13.882944 で、trajectory add-only features は採用しない。
- 2026-06-02: exp009_formation_surface_guide の Kaggle full CV を完了。selected `formation_knn_no_gr` は CV 14.558630、best は `control_exp003_no_gr` 13.882944 で、formation guide は採用しない。
- 2026-06-01: exp006_hard_well_router_diagnostic の Kaggle full train を完了。`router_diagnostic_well_tags.csv`、`router_condition_summary.csv`、`router_candidate_rules.csv`、`router_diagnostic_metrics.json` を Kaggle output から取得し、CV 13.936732 を記録。
- 2026-06-01: exp007_hard_well_router の Kaggle full train / inference / submission を完了。selected `hard_router_low_gr_guarded` は CV 13.921559、Public LB 12.675、submission ref `53254030`。
- 2026-06-01: exp008_gr_ncc_matcher の Kaggle full CV を完了。selected `gr_ncc_no_gr_multi` は CV 14.641514、`gr_ncc_all_multi` は 14.661017 でどちらも control より悪化した。
- 2026-06-01: exp005_gr_gate_recalibration の Kaggle full CV を完了。selected `gate_low_gr_strict_hard` は CV 13.936732、gating 系 CV 最良は exp004 再現 `control_exp004_low_gr_any_hard` の 13.932968。
- 2026-06-01: exp005_gr_gate_recalibration の Kaggle inference と提出を完了。ref `53249562`、public LB 12.579。
- 2026-05-31: exp003_residual_ablation の Kaggle full CV と提出を完了。`feature_no_gr_signal` は CV 13.882944、public LB 12.852。
- 2026-05-31: exp004_gr_gating の Kaggle full CV を完了。selected inference candidate は `gate_low_gr_coverage_hard`、CV 13.932968。
- 2026-05-31: exp004_gr_gating の Kaggle inference と submit-check を完了。`submission.csv` は形式 PASS。
- 2026-06-01: exp004_gr_gating を提出。ref `53247991`、public LB 12.730。
- 2026-05-31: exp002_drift_minimal を exp001_baseline から作成し、drift residual model を実装。Kaggle train full CV は 14.124569 で exp001 から改善。
- 2026-05-28: exp001_baseline を実装し、full CV、提出生成、提出形式検証まで完了。
- 2026-05-27: ROGII コンペ設定、公式データ説明、評価指標、検証方針を Markdown と `project.yml` に反映。
- 2026-05-26: リポジトリ構成とベースライン実験の初期設定を作成。
