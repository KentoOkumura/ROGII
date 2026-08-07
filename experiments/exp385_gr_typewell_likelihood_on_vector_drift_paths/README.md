# exp385_gr_typewell_likelihood_on_vector_drift_paths

## 状態

- ルート: `pf_beam`
- 状態: exp383 Stage 0 resource FAILにより未実装で閉鎖
- CV / Public LB / Private LB: -
- 作成日: 2026-07-24
- 親実験: `exp384_fault_aware_piecewise_stratigraphic_vector_field`

## 仮説

exp384のbase/component物理pathは、対象horizontal GRとtypewell GRの一致度で
識別できる。固定Student-t likelihoodとexact posterior平均により、
formation geometryだけのdomain posteriorを改善できる。

## 変更点

- base 1 + component最大8の最大9 candidateを固定する。
- candidate TVTでtypewell GRをhorizontal位置へ写像する。
- 256-row / stride 64のStudent-t (`df=4`) GR emissionを作る。
- `0.98 stay + 0.02 current prior`のexact forward-backwardを使う。
- primaryはhard top1ではなくposterior-weighted TVT。
- GR欠損/ineligible位置はexp384へexact fallbackする。

## 検証方針

- exp383/384 Stage 0/1 PASSと保存SHA一致が先行条件。
- Stage 0はknown-prefix rolling-originとreal-vs-circular GR識別。
- Stage 1はexp384比`>=0.50 ft`、4/5 folds、1000+/hidden-like改善。
- suffix true TVTとtarget生Formationはfreeze前read 0。

## 実行入口

- 正規Notebookはtemplate scaffoldで未実装。
- 実装、Kaggle package/push/run、推論、提出は先行条件PASS後の別承認。
- 773-well exact decoderもStage 0 PASS後の別承認。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- candidate pathを変更せず、GR/typewellの識別力だけを切り分ける設計にした。

### 悪かった点

- exp383がresource gateをFAILし、exp384候補bankも生成されないため、
  candidate diversityとGR eligible coverageは未確認のまま閉じた。

## リスク / 注意

- candidate diversity不足、typewell mismatch、GR registration、missingnessが主要リスク。
- posterior平均が非物理な中間pathになる可能性を別診断する。

## 次

- 実装、Kaggle run、inference、submissionは行わない。
