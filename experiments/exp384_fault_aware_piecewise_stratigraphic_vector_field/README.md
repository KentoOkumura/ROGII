# exp384_fault_aware_piecewise_stratigraphic_vector_field

## 状態

- ルート: `pf_beam`
- 状態: exp383 Stage 0 resource FAILにより未実行で閉鎖
- CV / Public LB / Private LB: -
- 作成日: 2026-07-24
- 親実験: `exp383_all_tvt_stratigraphic_vector_drift_field`

## 仮説

exp383のsmooth fieldが断層両側を平均して残す誤差を、outer-trainの地層面不連続と
正解TVT residualからpiecewise domainへ分ければ、物理pathをさらに改善できる。

## 変更点

- exp383の256 ft donor nodesから固定fault graphを作る。
- 6地層geometryと`S`/rate residualが同時に不連続なedgeだけを切る。
- domain別absolute/vector fieldを作る。
- targetではhard domainを選ばず、smooth base mass最低0.25のposterior平均にする。
- exp383のsurface/catalog/prefix/fallback/path solverは固定する。

## 検証方針

- exp383 Stage 0/1 PASSと保存SHA一致が先行条件。
- Stage 0はgraph/domain/posteriorをtruth前にfreeze。
- Stage 1はexp383比`>=0.50 ft`、4/5 folds、1000+/hidden-like改善。
- no-fault/unsupported位置はexp383へexact fallback。

## 実行入口

- 正規train Notebook:
  `exp384_fault_aware_piecewise_stratigraphic_vector_field_train.ipynb`
- 編集元:
  `exp384_fault_aware_piecewise_stratigraphic_vector_field_compact_selfcontained_train.py`
- inference Notebookは、別承認まで明示的にfail-closed。
- exp383 Stage 0/1 PASS manifestと全入力SHAを`config.yaml`へpinするまでrunできない。
- Kaggle package/push/run、科学score、推論、提出は親resource FAILにより無効。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- exp383の設定を変えず、fault domainの寄与だけを反証できる設計に分離した。
- fixed AND fault cut、stable component、component field、soft posterior、
  exact exp383 fallback、late truth joinをself-contained Notebookへ実装した。
- 専用test `14 passed`、Ruff、py_compile、Jupytext round-trip、
  strict experiment validationを通した。

### 悪かった点

- exp383 version 1はtruth join前に停止し、5-fold surface stage投影が
  固定runtime gateの3.59倍となった。必要な親PASS生成物とSHAは存在しないため、
  fault graphのcoverage、component数、runtime、CVは未確認のまま閉じた。

## リスク / 注意

- trainで見えるfault domainがtarget surface signatureから識別できない可能性がある。
- graph fragmentationとhidden fault分布差をStage 0/1で分離して報告する。

## 次

- exp384のpackage/push/run、Stage 0/1、inference、submissionは行わない。
