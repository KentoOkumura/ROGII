# 設計

## アプローチ

```text
exp263 primitive candidate cache (6 paths)
                +
non-TVT row context / source-native confidence
                ↓
15 pair differences D_ij = P_i - P_j
                ↓ 512-row fixed blocks
gap / slope-gap / curvature-gap / crossing / rank-persistence
                ↓
outer-train robust scaler + KMeans(K=3)
                ↓
outer-valid soft assignment + regime separability readout
                ↓ Stage 0 guard通過後・別承認
exp264 global fallback + 3 dual-objective soft experts
```

## 実験範囲

- 対象実験: `exp265_target_free_pairwise_candidate_divergence_soft_experts_on_exp264`
- Route: `ensemble`
- 親実験: `exp264_exp263_candidate_confidence_dual_selector`
- 候補親: `exp263_last_anchor_better_candidate_confidence_pair_cache`
- 変更する変数: 15 primitive-pair divergence fingerprint と outer-fold soft regime assignment。
- 固定する変数: exp263 の 6 primitive値、exp264のfold/candidate/objective、512-row block、K=3、seed 42。
- 初回実装: Stage 0 separability audit のみ。学習・inference・submissionは行わない。

## Pairwise fingerprint

primitive は `exp226_k16`、`selfgr_hmm_a070`、`likpf_mean`、`exact_hmm`、
`pf_ancc`、`beam_mean` の6本とする。全15 pairで、blockごとに次を計算する。

- signed gap: mean / end / std。
- absolute gap: mean / p90 / max。
- dynamics: gap slope、slope change、first-difference correlation。
- topology: zero-crossing count、sign persistence、divergence expansion ratio。
- scale-free context: block length、MD span、candidate rank switch count。

さらに6候補全体から bank median/IQR/max、centered candidate matrix の第1/第2 singular value比、
geometry-vs-GR/HMM contrast、PF-vs-Beam contrastを作る。absolute candidate TVT、last-known TVT、
true TVT は regime feature に含めない。

## Stage 0 orchestration

1. exp263 cache manifest、candidate order、fold/row/well coverage、SHAを検証する。
2. fold別の6 primitive value/confidence partitionを読み、well/rowでstrict alignmentする。
3. well内のevaluation rowを`well_row_idx // 512`で固定block化し、fingerprintを生成する。
4. outer foldごとに他4 foldsだけでmedian/IQR scalerとKMeans(K=3)をfitする。
5. outer-valid blockへ距離、soft probability、entropy、hard regimeを保存する。
6. centroidをgeometry/HMM/PF contrastの決定的順序でcanonical labelへ対応付けする。
7. assignmentをrowへ戻し、exp264 `candidate_score_oof.parquet`とID/candidate/foldでjoinして
   regime別 expected-error bias、within10 calibration、best candidate familyをreadoutする。

## Stage 1 条件付き設計

Stage 0 guard通過時だけ別承認で有効化する。3 regimes × 2 objectives × 5 outer folds =
30 CPU boosters。保存済みexp264 global score/modelをcontrolとし、親/control再学習は0。
expert mixtureはhard switchを使わず、`1 - normalized_entropy(gate)`でglobalからexpert mixtureへ
決定的に縮約する。Stage 1実装・Kaggle pushは本実験の初回scope外とする。

## 再現性設計

- seed policy: fixed seed 42。KMeansはfoldごとに`42 + outer_fold`、`n_init`固定。
- stochastic 処理: KMeans初期化のみ。global RNGを使わず、estimatorへ明示seedを渡す。
- PF/Beam / likelihood-PF / seed bagging: 再実行なし。exp263保存cacheをSHA固定入力にする。
- 並列処理と乱数: KMeansは単一process。feature aggregationはdeterministic sort/group順。
- CPU/GPU: Stage 0はKaggle CPU、0 booster。GPUなし、internet disabled。
- train cache: Parquet logical/schema SHAとmanifest SHAを記録する。
- model/prediction/submission: Stage 0では生成しない。centroid/assignment/fingerprint SHAを記録する。
- Kaggle bootstrap: source/package config、notebook、support `src/` SHAとmetadataをpush前に照合する。

## リスク

- リークリスク: regime fitやfeature選択にtrue TVT/error/oracleを混ぜるとrouting leakageになる。
- 相関リスク: 12 candidate全pairはblend由来の決定的重複が多いのでprimitive 6本に限定する。
- CV/LBリスク: exp007/018のhard routerはfold外で不安定だったため、global fallbackを外さない。
- sample fragmentation: regimeが小さい場合はStage 1へ進めずglobal exp264を維持する。
- cluster label switching: centroidを決定的keyでcanonicalizeし、fold間の対応を保存する。
- runtime/memory: full row×pair wide tableを常設せず、fold/well block集約後だけ保存する。
- exp264依存: Stage B artifactが未完成またはcontract不一致ならreadoutを停止し、推測補完しない。

