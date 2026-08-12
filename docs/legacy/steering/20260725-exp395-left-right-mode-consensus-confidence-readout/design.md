# exp395 設計

## 結論

逆方向の予測器は作らない。exp209 の同じ absolute-TVT exact-HMM grammar と、
exp391 の stable mode lineage を固定し、判定点の左右にある重ならない GR だけで
各 mode の evidence distribution を別々に計算する。

左右の mode posterior overlap を confidence として保存し、truth late-join 後に
「不一致ほど persistent offset / large error が多いか」だけを監査する。
予測値は一切変更しない。

## 実験範囲

- 対象実験: `exp395_left_right_mode_consensus_confidence_readout`
- Route: `pf_beam`
- 親実験: `exp391_prefix_anchored_mode_persistence_hmm_readout`
- decoder 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- primary prediction reference: exp263 fixed physical candidate
  `exp226_w500_50_50`、OOF `8.238331546 ft`
- 直接物理 reference: exp226 OOF `9.427109596 ft` / Public LB `9.837`
- 変更する変数: 観測 GR を左右の disjoint evidence へ分けた
  mode-confidence readout の追加だけ
- 固定する変数: exp209 emission / transition / grid / calibration、exp391 mode threshold /
  lineage matching、outer 5 folds、既存 prediction、全 window / gap / gate
- 生成しないもの: 新規 TVT candidate、補正後 prediction、submission

## 共通 mode identity

mode は row ごとの mass rank ではなく、exp391 と同じ次の contract で追跡する。

- posterior peak / basin threshold は exp391 の固定値をそのまま使う。
- prefix start-prior lineageをanchorとする。
- adjacent row間はmaximum transition-transport overlapでmode IDを対応付ける。
- rank swapはmode switchと数えず、cross-mode edgeとgradual driftはswitchと数える。
- merge / split / collisionでidentityを解けない場合は`unmatched`とし、
  高confidenceへfallbackしない。

exp391 Stage A1がHMM内部原因をeligible eventの60%以上かつ4/5 foldsで支持しない場合、
このmode carrierは不適格としてexp395を未実装のまま閉じる。

## 判定点と左右 evidence

primary scopeは、exp391 Stage A0でtruth-freeに固定済みの
1,234 decoder-separation eventsの中心点とする。secondary coverage readoutとして、
unknown suffixを256 rows strideで走査する固定checkpointも保存する。

各checkpoint `c`で次を固定する。

- left window: `[c - 64 - 512, c - 64)`
- right window: `[c + 64, c + 64 + 512)`
- window length: 512 rows
- exclusion gap: checkpointの左右64 rows
- boundary: 有効観測が各側256 rows未満ならprimary判定から除外し、理由を保存する

既知prefixから得るaffine GR calibration、absolute-TVT grid、transition grammarは
左右で共通とする。一方、左右のlikelihood計算に同じtarget GR rowを入れない。
したがってこれは完全な統計的独立ではなく、固定physics/calibration条件下の
disjoint-observation consistency testである。

同じmode lineage集合について、left / right log evidenceをそれぞれ正規化し、
`P_L(m)`、`P_R(m)`を得る。左右のmode数が異なる場合はunion上へ展開し、
存在しないmodeのmassを0とする。

## confidence

primary continuous confidenceは重みなしのposterior overlapとする。

```text
C_lr(c) = sum_m min(P_L(m), P_R(m))
D_lr(c) = 1 - C_lr(c)
```

合わせて次を保存する。

- `left_top_mode_id` / `right_top_mode_id`
- `left_right_top_mode_agree`
- mode center TVT差
- left / right top1-to-top2 log marginとそのminimum
- Jensen-Shannon divergence
- adjacent checkpointでのmode agreement persistence
- unmatched / boundary / normalization flag

primary error-risk scoreは`D_lr`だけとし、metricを見たweight付き合成は作らない。
high / low confidenceの固定絶対thresholdはこの実験では採用せず、事前固定した
quartile readoutとcoverage-risk curveで評価する。

## 物理・モデル間の二次vote

左右の誤ったalias一致を識別する補助として、confidence freeze後・truth join前に
次を同じmode basinへ割り当てる。

- target GRを使わないexp226 `tvt_geop`: physical third vote
- LikPF、exact HMM、exp226 final、exp263 fixed candidate: cross-model report

primary confidenceとprimary scientific gateには混ぜない。`tvt_geop`を含む三者一致、
PF/HMM/exp226の一致率、unmatched率を二次表として報告するだけとする。

## negative control

right GR windowをwell内で固定`2048 rows` circular shiftしたnullを1つだけ作る。
shift不能の短いwell/eventはnull対象外として理由を保存する。shift量、window、
gapは同一runや結果確認後に変更しない。

real `D_lr`がcircular nullよりlarge-error riskを識別できることを要求する。
mode label permutation、追加shift、window gridによる救済は行わない。

## Stage 0

2026-07-25にexp391 Stage A1がFAILしたため未実行で閉鎖。以下は実行しなかった
事前固定contractの履歴として残す。

- diagnostic variant: 1
- exact-HMM well runs: 16
- reporting folds: 5
- LightGBM config / trained fold / booster: 0 / 0 / 0
- PF / Beam / GPU: 0 / 0 / 0
- parent/control rerun: 0

必須gate:

- 16/16 wells完走、finite left/right evidence coverage 1.0
- exp209 full smoother posterior mean parity `<=1e-5 ft`
- posterior normalization error `<=1e-8`
- mode ledger duplicate / identity collision 0
- eligible primary events `>=10`
- confidence / mode / input SHA freeze前のtruth/error/hidden-like read 0
- projected full runtime `<=30,600 sec`
- projected peak RSS `<=25 GB`

Stage 0ではerror AUCやRMSEを計算しない。PASSしてもfull OOFは別承認とする。

## full OOF scientific readout

別承認後だけ1 diagnostic variant / 773 exact-HMM well runsを実行する。
confidence tableとlogical SHAをfreezeした後にsuffix truth、error、hidden-like roleを
late joinする。

primary scopeはevent-centered checkpointとし、次を全ANDで要求する。

- `D_lr`による`abs(error) > 10 ft` pooled AUC `>=0.60`
- 5 folds中4 folds以上でAUC `>0.55`
- low-confidence quartile RMSE - high-confidence quartile RMSE `>=2.0 ft`
- top-mode disagreement群 / agreement群のbad10 rate ratio `>=1.5`
- eligible events `>=500`、eligible wells `>=300`、全5 foldsをcoverage
- real AUC - circular-right-null AUC `>=0.03`
- 1000+、hidden-like spatial、hidden-like typewell-purgedの3 scopeで
  risk directionが一致

RMSE prediction自体はexp209 / exp226 / exp263の保存値を報告するだけで、
exp395のCV improvementは定義しない。

gate FAIL後のthreshold、window、gap、mode matching、confidence formula、
physical vote weight、fallback、selector救済は禁止する。

## リークリスク

- event / checkpoint、mode lineage、confidence、negative controlをtruth前にfreezeする。
- outer-valid/test suffix truth、error、raw formation、hidden-like roleはfreeze前に読まない。
- exp226 `tvt_geop`は保存済みgroup-safe outer-fold contractを要求する。
- Public LBはmode、window、gateの選択に使わない。
- cross-model prediction一致をprimary scoreへ混ぜない。

## 再現性設計

- seed policy: RNGなし。fold / well / row / checkpoint / mode / direction順を固定する。
- stochastic処理: なし。
- PF/Beam / likelihood-PF / seed bagging: 新規実行なし。
- 並列処理: CPU、固定thread、stable reduction。global RNGなし。
- runtime: Kaggle private CPU、internet offを想定。実行は未承認。
- input SHA: exp391 event manifest、exp209/226/263入力のlogical/decompressed SHAを固定。
- output SHA: checkpoint、mode ledger、left/right evidence、confidence、null、
  late-joined metric tableのlogical/content SHAを分ける。
- model manifest: fitted modelなし。decoder/scientific contract SHAを代替記録する。
- prediction/submission SHA: predictionとsubmissionを生成しないため対象外。
- deterministic anchor: 同一contractのrerunで主要logical SHAが一致するまでfalse。

## 主なリスク

- 左右が同じ反復GR aliasを支持し、誤modeでも高confidenceになる可能性がある。
- exp391のmode carrierがHMM固有で、PF/exp226のmodeを完全には表さない。
- 512-row windowが境界eventを跨ぎ、mode evidenceを平均する可能性がある。
- GR calibrationとtransition priorを左右で共有するため、agreementを過大評価し得る。
- exp391 Stage A1がFAILした場合は実装前に停止し、HMM modeを使う本設計を救済しない。

## 実装状態

design-only。template scaffold以外のNotebook source、helper、test、package、
Kaggle kernel、生成物は作成していない。
