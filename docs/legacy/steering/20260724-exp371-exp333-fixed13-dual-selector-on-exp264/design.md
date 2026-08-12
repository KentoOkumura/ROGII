# 設計

## アプローチ

corrected exp264 Stage C v6 の candidate-long dual selector をそのまま使い、
exp263 の12候補へ `exp333_segment_offset` を13本目として追加する。

exp333 OOF は `well_id,row_idx,outer_fold,tvt_pred_stage1` の allowlist だけを読み、
同居する `tvt_true` は pre-freeze に開かない。`outer_fold`はsaved-exp226 OOF
provenanceとして監査にのみ使う。exp333全行を`well_id,row_idx`でglobal index化し、
exp263 の各 selector fold bundleへkey joinして`float32`の13列目へ連結する。
source foldをselector特徴へ渡さず、source/selector 5×5 overlapと全行coverageを保存する。
source-native confidence は持たないため `confidence_valid=false` とし、全候補共通の
candidate value、anchor distance、local shape、bank disagreement、candidate/family/kind
one-hot だけを使う。

selector は exp264 と同じ2目的を学習する。

- `pred_abs_error`: candidate absolute error の L1 回帰
- `p_within10`: absolute error が10 ft以内かの二値分類

outer 5-foldの各 outer-train 内を deterministic well-balanced inner 4-foldへ分ける。
outer-train compact は inner OOF、outer-valid compact は4 inner model ensembleだけから
生成する。Stage A feature auditは同じKaggle runの fit 前に実行し、schemaを凍結してから
Stage Cへ進む。

## 実験範囲

- 対象実験: `exp371_exp333_fixed13_dual_selector_on_exp264`
- Route: `ml_model`
- 親実験: `exp264_exp263_candidate_confidence_dual_selector`
- candidate source: exp263 corrected deployable12 + exp333 saved Stage 1 OOF
- 変更する変数: candidate count `12 -> 13`、primary domain `11 -> 12`、
  candidate ID/family/kind encoding、compact feature count `74 -> 77`
- 固定する変数: selector 2目的、LightGBM parameters、outer 5 / inner 4、
  exp263 selector fold identity、sampling、raw context、shape windows `[32,128,512]`、
  fixed fallback domain 7本、candidate order先頭12本
- Stage A/Stage C run: 1 variant、2 objectives、5 outer × 4 inner、
  40 CPU boosters、parent/control retraining 0
- 明示例外で進める後段: fixed13 compact 77列を clean273 へ add-only、
  3 configs × 5 folds = 15 T4 GPU boosters、保存済み exp264 Stage D v3 OOF を
  comparison control、control retraining 0。元のStage C safety gate FAILは保持する。

## Stage D add-only設計

- surface: 監査済み`exp218 clean273` + saved `fixed13 compact77` = 350 features
- variant: `selector_compact_addonly`のみ
- model: exp218 / exp063 LightGBM familyのconfig index `[0,1,2]`
- fold: exp371 Stage Cのdownstream outer 5 fold
- compute: `1 × 3 × 5 = 15` T4 GPU boosters
- runtime: internet off、`gpu_use_dp=true`、`deterministic=true`、
  `force_col_wise=true`、threads 8
- comparison: saved exp264 Stage D v3 parent12 compact add-only
  `8.460811237612477`。保存済みcontrolは再学習しない。
- gate: pooled改善、3/5 folds、near/1000+/hidden-likeの最大悪化`+0.02 ft`、
  by-well p95/worstの最大悪化`+0.25 ft`
- scope外: Stage C再学習、candidate/PF/HMM再生成、current-test inference、submission
- 例外の意味: selector平均改善を下流で検証する許可であり、Stage Cのby-well
  p95/worst不合格を覆さない。

## 再現性設計

- seed policy: exp264 と同じ seed 42。row sampling は immutable stage/fold key の
  SHA256 seedを使い、inner splitはwell row count + well id stable tie-breakで固定する。
- stochastic 処理の有無: LightGBM の row/column samplingのみ。global RNGをworker内で使わない。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。保存済み exp263 candidateを読む。
- 並列処理と乱数の関係: samplingをfit前に決定し、LightGBMは
  `deterministic=true`, `force_col_wise=true`, `n_jobs=8`。
- CPU/GPU runtime と deterministic flags: Stage A/CはKaggle CPU、internet/GPU off。
  GPU Stage Dは2026-07-24の明示例外承認で有効化する。GPU bitwise一致は主張せず、
  入力、モデル、OOF、出力artifactのSHAを保存する。
- train cache / test feature regeneration の SHA 記録方針:
  exp263 manifest/catalog、exp333 OOF file/decompressed、candidate contract、
  feature schema、compact schema、outer-valid score、25 compact partitionを記録する。
  exp333 current-test candidate decompressed SHA
  `7571c6281bd2ab484e7bf536a876b8072407b272a0ef0ec5112ca06897a717cd`
  は inference port用の将来契約として記録するが、今回のtrain入力には使わない。
- model manifest / prediction / submission SHA 記録方針:
  40 selector model SHA、nested model manifest SHA、outer-valid candidate score SHA、
  compact manifest/partition SHAを記録する。submissionは生成しない。
- Kaggle package bootstrap 確認方針: canonical notebook source、config、
  candidate contracts、shared pipeline、exp333 loaderをbootstrap ZIPから再抽出し、
  source側SHAと一致させる。run approvalと40 boosters/control 0も埋め込みconfigで再確認する。

## リスク

- リークリスク: exp333 gzipにtruthが同居する。pre-freezeはusecols allowlistで読み、
  fold/key/candidateをfreezeした後の評価にだけraw truthを使う。outer-valid wellは
  inner split/fit/early stoppingから除外する。exp333 predictionはsaved-exp226 source
  foldで各well自身に対してOOFだが、独立生成されたexp263 selector foldとは一致しない。
  親exp264 candidate bankと同じglobal key join semanticsを採用し、この差を5×5
  provenance auditとして明示する。source fold自体はmodel featureへ入れない。
- CV/LB 不一致リスク: exp333 directはnear/worst gate FAIL、exp264 hard selectorもfixed fallbackより
  悪い。pooledだけで昇格せず、fold/near/1000+/hidden-like/worst/candidate usageを固定gateにする。
- ランタイム/メモリリスク: 13候補化でcandidate-long rowsが12候補比8.33%増える。
  chunk化を維持し、40 CPU boosterの見積りは exp264 Stage C v6の約3,443秒を基準に
  約1時間強とする。downstream GPUは同じrunに含めない。
- 再現性リスク: pandas round-trip logical hashはexp361 v1で不安定だったためhard contractにしない。
  exp333はfile/decompressed SHAとpost-read content SHAを分けて記録する。
- 選択不能リスク: exp333がtop1にほぼ使われない可能性がある。使用率gateを事前固定し、
  FAIL時に同じOOFでthreshold/weight/domainを救済しない。
