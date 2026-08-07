# 設計

## 仮説

exp226のoverall RMSEは大局的offset/driftに引っ張られているが、`tvt_geop + gr_delta`の局所形状は、
exp293 deployable12より強い可能性がある。各blockで真値との差の定数成分と一次傾向を診断上だけ商空間へ落とし、
残差形状を比較すれば、局所sourceとして後続hybridへ使う根拠を直接検証できる。

## 実験範囲

- 対象実験: `exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit`
- Route: `pf_beam`
- 親実験: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- 比較: `exp293_physics_only_candidate_bank_headroom_contract`の固定deployable12
- 変更する変数: 評価時にblockwise offset-only / affine nuisanceを除いたlocal-shape readoutだけ
- 固定する変数: 候補値、候補順、fold、row、block、horizon、scope、tie、閾値、入力SHA
- 対象外: Lateフェーズ、候補生成、補正、selector、decoder、モデル学習、推論、提出

## 対象成分

```text
P_geop  = tvt_geop
P_preU  = tvt_geop + gr_delta
P_postU = tvt_pred = exp293 candidate exp226_k16
```

`P_preU`をprimary local sourceとする。`P_postU`は`exp226_k16`のaliasであり、rank上の重複候補にはしない。
`P_geop`と`P_postU`は、GR correctionとU projectionが局所形状へ与える影響のpaired diagnosticに限定する。
truthを見てprimary componentを差し替えない。

## blockwise quotient

exp293のH128/H256/H512非重複block割当をそのまま使う。候補path`p_i`、真値`y_i`、block`b`で
`e_i = y_i - p_i`とする。

- offset quotient: `q0_i = e_i - mean_b(e)`。
- affine quotient: block内row位置を`x_i in [-1,1]`へ正規化し、OLSで
  `e_i = a_b + b_b x_i + q1_i`をfitした`q1_i`。
- primary metric: 全rowを集約したH256/H512の`RMSE(q1)`。
- secondary: H128/whole-wellの`RMSE(q1)`、H128/H256/H512の`RMSE(q0)`、block内一階差・二階差error、
  block win/unique-best、fold、1000+、hidden-like 2面、by-well p50/p95/worst。

`a_b,b_b`はtrue suffix TVTを使うoracle nuisanceなので、metric集計中だけ保持し、係数、補正path、選択pathを
artifactへ書かない。この値を後続candidate、feature、補正、thresholdへ使わない。

### singleton final block policy（2026-07-20承認改訂）

exp293固定block assignmentには長さ1の最終blockがH128/H256/H512で`4/2/2 wells`存在する。1行では
interceptとslopeを識別できないため、block境界は変えず、selected row数2以上だけをaffine-eligibleとする。

- 長さ1はaffine RMSE/rank/block win/strict unique-bestの分母から除外する。
- technical affine coverageは`valid affine-eligible rows / affine-eligible rows == 1.0`で判定する。
- singleton block/row/well数をhorizon・scopeごとに記録する。
- 長さ2以上でdeterminant不良またはnonfiniteとなるblockは従来どおりtechnical FAILで、fallbackしない。
- offset-only secondary readoutは変更しない。singletonのoffset residualは定義可能だが、affine superiorityの
  根拠やunique-bestには使わない。
- exp293のblock ID、boundary、SHAは変更しないため、新しいboundary探索やhorizon救済には当たらない。

## freeze順序とリーク防止

1. exp226 OOFは`well_id/fold/row_idx/suffix_offset/tvt_geop/gr_delta/tvt_pred`だけをallowlistで読む。
   同じ物理fileにある`tvt_true/error/abs_error`はmaterializeしない。
2. exp226 OOF decompressed SHA、exp293 bank content SHA、block assignment decompressed SHAを照合する。
3. `P_preU`をfloat64演算で再構築し、`P_postU == exp226_k16`の最大差`<=0.001 ft`を確認する。
4. row/fold/well/component/candidate/block/tie/scope/threshold manifestをtruthなしでcontent freezeする。
5. freeze SHAの確定後だけraw trainからtrue suffix TVTを別loaderで読み、identity完全一致でjoinする。
6. quotient readoutを計算し、係数と補正predictionを保存せず集計値とSHA evidenceだけを保存する。

freeze前にtruth列がmaterializeされた場合、identity/SHAが不一致の場合、finite coverageが1.0でない場合は
scientific metricを出さずtechnical FAILとする。

## PASS / FAIL

technical guard全通過に加え、`P_preU`が次をすべて満たした場合だけPASSとする。

technical affine coverage 1.0は全rowではなく、上記のaffine-eligible rowsに対して要求する。singleton除外は
全候補へ同一に適用し、候補別・truth別の除外を認めない。

- H256/H512 pooled affine-quotient rankがそれぞれ3位以内。
- H256/H512の少なくとも一方で1位。
- 各primary horizonで5 folds中4 folds以上が3位以内。
- H256/H512の両方で`P_postU`に非悪化。
- H512の1000+、hidden-like spatial、hidden-like typewell-purgedがそれぞれ3位以内。
- H256/H512の少なくとも一方でstrict unique-best block比率`>=0.05`。

PASS時に許可するのは`downstream_branch_contract.md` Stage 2の独立実験化だけである。FAIL時は本枝を閉じ、
component、horizon、quotient、scope、平滑化、weightの救済gridを作らない。

## 後続2・3・4

設計の正は実験配下`downstream_branch_contract.md`とする。

- 2: fixed `S512`で`P_preU`のlocalとexp293各候補のglobalを組み、原12 + hybrid12の24候補bankを監査する。
- 3: Stage 2 PASS時だけ、24候補×registration×reliabilityのsemi-Markov posterior meanを作る。
- 4: Stage 2 supportが強いのにStage 3が不足した場合だけ、明示承認後にouter5×inner4 nested rankerを
  semi-Markov unaryとして追加する。hard top1やdirect correctionにはしない。

exp295 candidate-free learned SSMと、exp293/exp297 fixed12 branchは独立のまま維持する。

## 再現性設計

- seed policy: RNGなし。fold/well/row/candidate順とfloat64 reduction順を固定する。
- stochastic処理: なし。
- PF/Beam / likelihood-PF / seed bagging: 再生成なし。保存済み候補を読むだけ。
- 並列処理: 初回実装は`num_workers=1`。global RNGを使わない。
- runtime: Kaggle private CPU、GPU/AMP/internet offを想定するが、実行は別承認。
- SHA: exp226 input、exp293 bank、block assignment、component manifest、pre-truth freeze、truth readout、各artifactを記録する。
- gzip: raw file SHAとdecompressed content SHAを分け、後者を主証拠にする。
- model/prediction/submission SHA: model、deployable prediction、submissionを生成しないため対象外。
- deterministic anchor: fixed-input diagnosticでありsubmission anchorではない。

## リスク

- リークリスク: oracle quotientが良くても予測時にはoffset/slopeを知らない。したがって絶対RMSEではなく
  局所sourceの存在証明にしか使わず、係数/補正pathを保存しない。
- 重複リスク: `P_postU`と`exp226_k16`を二重候補にするとrank/choiceを歪めるためalias扱いにする。
- 局所尺度リスク: H128だけの改善を長距離signalと誤認しない。primaryをH256/H512へ固定する。
- CV/LB不一致: exp298は提出可能scoreを作らず、LBを推定しない。
- ランタイム/メモリ: 3,783,989 rows × component/candidate × 3 horizonsのblock集計が主。row-wise oracle predictionは保持しない。
- 再現性: source artifactやblock SHAが一致しない場合は再構築で推測せず停止する。
