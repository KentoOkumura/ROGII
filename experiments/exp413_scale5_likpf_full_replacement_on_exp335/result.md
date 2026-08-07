# exp413_scale5_likpf_full_replacement_on_exp335 結果

## 状態

Stage 0 replacement preflightとStage C strict-nested selectorはKaggle
version 3、Stage S signed-residual selectorはversion 1で完了した。
Stage Cはscore guardとleakage audit、Stage Sはtechnical / score gateをPASS。
Stage D downstreamはversion 2で15/15 GPU modelを完了し、固定technical /
primary gateをPASSした。final TVT CVは`7.884803`。current-test推論version 3は
公開データの14,151行 / 3 wellsを固定assertしていたため、code submission
ref `55078306`のhidden rerunで失敗した。固定assertだけをsample由来の動的な
row / ID / nonempty-well契約へ置換したKaggle CPU version 4は完了し、
`/kaggle/working/submission.csv`を生成した。公開predictionはversion 3と完全一致し、
取得後submit-checkもPASSした。ユーザー実施のversion 4 code submission
ref `55080377`は`COMPLETE`、Public LB `7.201`となった。

## 仮説

exp404の`likpf_scale_5_x1p0`はexp072互換`likpf_mean`よりdirect OOF RMSEを
約`0.680376 ft`改善した。このprimitiveをexp335の固定12候補と全依存特徴へ
一貫して全面置換すれば、現行Public-LB referenceを改善できる。

## 設定

- 親: `exp335_signed_residual_meta_on_exp264`
- 変更: `likpf_mean` semantic slotを`likpf_scale_5_x1p0`へ全面置換
- 候補: 12のまま。5 slot再計算、7 slot固定
- 特徴: rebuilt clean273 + compact74 + signed23 = final370
- 学習実績: 40 CPU + 20 CPU + 15 GPU = 75 boosters
- control再学習: 0
- 検証: 親と同じwell outer 5 / selector inner 4
- メトリック: RMSE
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| Stage 0 technical gate | PASS |
| Stage 0 rows / wells / partitions | 3,783,989 / 773 / 5 |
| changed / unchanged candidate | 5 / 7 |
| changed candidate rows | 3,698,554 |
| unchanged candidate max abs error | 0.0 ft |
| formula parity max abs error | 0.0 ft |
| parent old mean parity max abs error | 0.0 ft |
| Stage C technical / score / leakage gate | PASS / PASS / PASS |
| Stage C models | 40 |
| Stage C compact partitions / rows | 25 / 18,919,945 |
| Stage C outer-valid score rows | 45,407,868 |
| expected-error MAE / prior | 3.720634 / 5.700200 |
| within10 logloss / prior | 0.349579 / 0.499814 |
| within10 Brier / prior | 0.108064 / 0.160703 |
| 3 score metricsの改善fold数 | 5/5 / 5/5 / 5/5 |
| Stage S technical / score / total gate | PASS / PASS / PASS |
| Stage S models / compact partitions | 20 / 25 |
| Stage S compact / outer-valid score rows | 18,919,945 / 45,407,868 |
| signed-residual pooled RMSE / prior | 8.291963 / 10.854996 |
| signed-residual pooled improvement | 2.563032 ft |
| signed-residual改善fold数 | 5/5（要件4/5） |
| candidate別改善数 | 11/12 |
| Stage D technical / primary gate | PASS / PASS |
| Stage D models / unique model SHA | 15 / 15 |
| saved exp335 RMSE | 8.146108 |
| replacement RMSE | 7.884803 |
| pooled gain | 0.261305 ft |
| nonworse folds | 5/5（要件3/5） |
| 最大scope delta | -0.019498 ft（上限+0.02 ft） |
| by-well delta p95 / worst | +1.228715 / +9.033462 ft（report-only） |
| +1 / +3 / +5 ft悪化well数 | 55 / 8 / 6 |
| control再学習 / PF well-runs | 0 / 0 |
| current-test inference | version 4 COMPLETE / output validation PASS |
| inference rows / wells | 14,151 / 3 |
| saved selector / signed / TVT models | 40 / 20 / 15 |
| current-test scale5 changed rows | 14,093 / 14,151 |
| scale5 abs/delta / candidate formula / signed top1 parity | 0.0 / 0.0 / 0.0 ft |
| inference runtime / log max timestamp | 432.680 / 462.250 sec |
| submission.csv / submit-check / user submit | 生成済み / PASS / ref 55080377 COMPLETE |
| code submission ref 55078306 | version 3 hidden rerun error（固定cardinality assert） |
| version 4 public parity | version 3と完全一致 |
| CV | 7.884802794404715 |
| Public LB | 7.201 |
| Private LB | - |

## 再現性

- deterministic anchor: false
- seed policy: exp404 train生成物再利用、current-testはstable per-well seed
- frozen input guard: raw / decompressed / logical / schema SHAを実装済み
- feature lineage: replacement primitive、candidate12、clean273、selector88、
  compact74、signed23、final370のschema/content manifestを実装済み
- Stage 0 kernel: `kentookumura/exp413-scale5-likpf-replacement-preflight`
  version 3、id_no `128773100`
- Stage 0 executed config SHA:
  `c06d0468bdb2798628989d362ad6729ab3bacafd78f3b883e5bccf088ff6f43a`
- Stage 0 executed source SHA:
  `470f04d97a840bda3535a46888a39c7d0116f8acd2fdbab64643374ac3aaded8`
- Stage 0 preflight SHA:
  `c1c536daaa4d0250578ba427882745f131223ed988f7b3adc812ed7daa33b258`
- semantic manifest file / logical SHA:
  `b3c98af2198b92756ac1db342e34bcf7bfc31ee54d9b2d85e5a01c0141fd34c7` /
  `8b2ff389467abb2480eeb00d55e8898fa777dbda5c0523b35ac7709bff425fdf`
- Stage C kernel: `kentookumura/exp413-scale5-likpf-selector-train`
  version 3、id_no `128776527`、runtime `6378.321 sec`
- Stage C executed config / source SHA:
  `a84c088da829584ab097bb5dc8ecd883362e2c8eb909f6ff1c3733c594369e1e` /
  `5757eef8c0aea3ad52e5f3538a44146aec494ea0b393c192c59b423a8ac478e6`
- Stage C model / compact manifest SHA:
  `7badcf45bccab0e2b0535ab7e7171e4c6cb7bf81e0bb115d5e283b1486b7b875` /
  `507429faa4fbc336dbc00e8edfee5a45788b8a58dbc2e15a440d5e7780d5f07f`
- outer-valid candidate-score SHA:
  `016408a6e77a3708be5cec285976b95b9f178921a51cac79780b169111c242cd`
- Stage C lineage / reproducibility manifest SHA:
  `2e0a36a7383d45a2f2aae7bba21c8160abff4a9d47e47a4d814300e729774aa1` /
  `ecf4b9a1e41199705438cf44982f417a4055a23f33e8b29e0d44c7b03093441a`
- Stage S kernel: `kentookumura/exp413-scale5-likpf-signed-train`
  version 1、id_no `128832243`、runtime `2984.194 sec`
- Stage S executed config / source SHA:
  `532372898a2a5cff0cbd99b817f82f71f0423626d79b607193cbbcbc8cca92eb` /
  `5757eef8c0aea3ad52e5f3538a44146aec494ea0b393c192c59b423a8ac478e6`
- Stage S model / compact / partition manifest SHA:
  `84a0047667ce9209ce63e3b9e935ff6c379d2d38b1ae4262318db94a47aaca9f` /
  `7a4282a25d7e7887e314cd3d01b8a09c81ff91ba3c1b1cf62e3197079ac93323` /
  `893d144fa1956b0cdcc4f4d75b9cfe3ebc05fd5bc54df0e7345a47f4b2c70ff1`
- signed outer-valid score / Stage S lineage / reproducibility manifest SHA:
  `004b0830fdc1a7893e5fdac77a217d717c6f7673a36065f50d297fa44c426841` /
  `d0e279e2c844d557c731107a0925f2d1c27de77eb64cfad9a498ab0b10ea0c13` /
  `6acebb4461ac2df6f94c6edf1068a0ba9e42bc8c8877d83b4b3ccfa586a3e7c1`
- feature content SHA: replacement `likpf_mean` 5 partition、Stage C compact
  25 partition、Stage S signed compact 25 partition、Stage D clean273 /
  final370をmanifestへ記録済み。
- Stage D kernel: `kentookumura/exp413-scale5-likpf-downstream-train`
  version 2、id_no `128914549`、runtime `17386.338 sec`
- Stage D executed config / notebook source / helper SHA:
  `67bea06a6082892cb93c6ac4e9be6dfe15665d48caca34c6b0da7878bf5a1c0e` /
  `5757eef8c0aea3ad52e5f3538a44146aec494ea0b393c192c59b423a8ac478e6` /
  `3b557b3a19c5dcc62dfe6567bfe4fdf6b47f3d2691549b87671d03763d0da566`
- final TVT OOF / model manifest / reproducibility manifest SHA:
  `9bd2d17778b3b27d771b12cbff72def8b87e6cdf14062e1c0ba192434cef4a9d` /
  `4b4f988154468ba6697cdd57c0a0c6bf7cc631e7b2bbe1f15fa8f51fdeb7c3df` /
  `5cfcf6d5fb76bd1b23782016967d95c4de86ac82489ab6f6f95a86bd41c1e472`
- current-test inference kernel:
  `kentookumura/exp413-scale5-likpf-current-test-inference`
  version 4、id_no `128975306`、runtime `432.680 sec`
- current-test executed config / source SHA:
  `7b99d4533b86966e7004b1dd401cdd0f628dff755441aa15239b4e4cf0ffef03` /
  `0f6fc81e56556aa6db828584ab2a2e58dde9db9cc4b54d6c12fa60e1c68f1388`
- prediction file / decompressed content SHA:
  `52ffb49110673f90b9b83b2e296e09b4ad0839164eda9ec13a91859937ebf136` /
  `875a1334ae3c90f841414f8f98d8877fb06234e17e0fd0b8d46385170a584dc4`
- inference metrics / reproducibility / log SHA:
  `8c26c5035c6b422738b351dd403674aa077bae5f838249f7e8ac5755943b3847` /
  `8c26c5035c6b422738b351dd403674aa077bae5f838249f7e8ac5755943b3847` /
  `3b4f2cbe105770adf5bbf2a64290062a667d87c19460c147c576b8551c600090`
- Kaggle version 4 submission SHA:
  `e9bb6bca7e19a087997c1f8d1d708d8ba0af21e770f5e44e1f1a52078142772f`。
  version 4のfull inference Notebook自身が`/kaggle/working/submission.csv`を
  生成し、取得後のsample互換・ID順・finite・source parity検証をPASSした。
  公開prediction / submissionはversion 3と完全一致した。
- rerun result: Stage 0 version 1/2の監査dtype不一致を修正してversion 3でPASS。
  Stage D version 1のPython 3.12 dynamic-import互換性不備を科学条件不変で修正し、
  version 2でPASS

## 解釈

Stage 0ではfrozen exp404入力の4種SHA、親cache 3,783,989行、
固定12候補の5 changed / 7 unchanged、formula、selector88 probe、
compact74 schemaが一貫していることを確認できた。version 1/2の失敗は科学値の
不一致ではなく、float32保存値を監査側だけfloat64理想式と比較したdtype契約の
問題で、許容幅を緩めずcache精度・演算順序のexact parityへ直した。

Stage C selectorはexpected-error MAE、within10 logloss、within10 Brierを
prior比で全5 folds改善し、replacement候補面のscore品質は支持された。
outer-trainはinner OOF、outer-validは4 inner model ensembleで生成し、
well分離と25 partition契約もPASSした。hard selector readoutは設計どおり無効で、
このStageだけではfinal TVTの改善を意味しない。

Stage S signed-residual selectorもpooled RMSEをcandidate別outer-train mean
priorから`2.563032 ft`改善し、5/5 folds、technical gateをPASSした。
20 modelはouter 5 × inner 4を一意に網羅し、signed23の25 partitionも
lineage/SHA契約に一致した。候補別では11/12を改善した一方、
`exp226_w500_50_50`はprior比`0.123613 ft`悪化した。これはStage S固定gateを
止める条件ではなく、Stage Dではfinal TVT overallと全固定scopeで吸収した。
ただしby-well tail悪化は残ったため、robust promotionとは分けて扱う。

Stage C version 1は40 booster完了後のreproducibility manifest seed欠落、
version 2はNotebook埋込configのstale run flagにより0 boosterで停止した。
version 3ではStage 0/selector契約SHAをrun前にseedし、埋込bootstrapを実展開監査して
完了した。科学条件は3 versionで変更していない。

Stage Dはsaved exp335の`8.146108`から`7.884803`へ`0.261305 ft`改善し、
全5 foldsがnonworse、near / mid / 1000+ / hidden-like 2面もすべて改善したため、
固定primary gateをPASSした。特に1000+は`-0.296706 ft`、spatialは
`-0.487557 ft`、typewell-purgedは`-0.501682 ft`だった。一方、report-onlyの
by-well tailはp95`+1.228715 ft`、worst well `fa31da94`で`+9.033462 ft`と
悪化している。したがってLB-oriented inference候補には昇格するが、
robust promotionとは扱わない。

Stage D version 1は学習前のPython 3.12 dataclass dynamic import互換性で停止し、
0/15 boosterだった。moduleを`sys.modules`へ登録してから実行するloader修正と
回帰testだけを追加し、variant、feature、fold、LightGBM config、gate、実行量を
変えずversion 2を完了した。

current-testでは3 wellsごとに`SHA256(likpf::test::<well>)`由来のstable seedを
固定し、500 particles ×128 seedsの同一trajectory bankからtemperature-5を
再生成した。14,093/14,151行でarithmetic meanから値が変わり、
absolute/delta roundtripは0.0 ft。Stage C / S / Dの40/20/15 model manifestと
全model SHAを照合し、新規学習0でfinal370へCPU推論した。予測はsample submissionの
ID/orderと完全一致し、重複・NaN・infなし、15 component平均との整合もPASSした。

inference version 1はparent-onlyのStage S parity toleranceをexp413 configから
参照した`KeyError: guards`で、最終model予測前に停止した。SHA固定済み
parent exp335 configを読む1行だけを修正し、科学条件不変のversion 2で完了した。
version 3はKaggle Notebook outputとしてsubmissionを生成したが、公開testの
14,151行 / 3 wellsをruntime固定assertしていたため、code submission
ref `55078306`のhidden rerunで失敗した。version 4はこのassertだけを
sample_submissionの行数・ID集合とnonempty well検査へ置換した。公開出力は
version 3と完全一致し、モデル・特徴・seed・予測計算は変更していない。

version 4 code submission ref `55080377`はPublic LB `7.201`で完了した。
親exp335 `7.517`から`0.316`、従来のensemble route anchor exp082 `7.601`から
`0.400`改善しており、ML routeの新しいPublic-LB referenceとなる。一方、
train-side by-well tail悪化は残るため、このLB改善だけでrobust promotionへ
再分類しない。

## 次

hidden互換version 4のcode submission ref `55080377`はPublic LB `7.201`で完了した。
今後はexp413をML Public-LB referenceとして比較に使い、train-side robust評価は
固定tail readoutと分離して維持する。
