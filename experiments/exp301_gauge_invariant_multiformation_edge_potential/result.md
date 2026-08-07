# exp301_gauge_invariant_multiformation_edge_potential 結果

## 状態

Kaggle private CPU version 2完了。Stage 0 support guard FAILのため、Stage 1を実行せずbranchを閉じた。

## 仮説

6 formationの井戸内edge差分はwell datumとformation topに対してgauge-invariantであり、共通2D potentialとして
積分すればknown prefix anchorから新しいdirect TVT candidateを生成できる。

## 設定

- 親: なし。比較anchorはexp226、独立性比較はexp289、candidate novelty比較はexp293。
- 検証: exp226 fold identityによる5-fold outer group CV。
- メトリック: direct RMSEとexp293 fixed12へのH512 add-one oracle headroom。
- シード: RNGなし。stable SHA splitだけを使う。
- 詳細: `config.yaml`とsteering `design.md`。

## 結果

| メトリック | 値 |
| --- | --- |
| Stage 0 identity | PASS。formation最大RMSE `0.008132852 ft`、median6最大RMSE `0.007869666 ft` |
| Stage 0 query component donor coverage | FAIL。fold `0.986729 / 0.979238 / 0.979066 / 0.969525 / 0.995853`、pooled `0.982164` |
| Stage 0 active component donor coverage | FAIL。fold `0.96 / 0.92 / 0.92 / 0.96 / 0.98` |
| Stage 0 leakage/runtime/row identity | PASS |
| Stage 1 solver fits | `0`（fail-closed） |
| Direct pooled OOF RMSE | 未生成 |
| Direct fold/subgroup/well guard | 未生成 |
| H512 add-one oracle novelty | 未生成 |
| Public / Private LB | 対象外 |

## 再現性

- deterministic anchor: false。単一完了runのみでrerun一致は未確認。
- seed policy: RNGなし、stable sort、SHA256 inner split。
- kernel: `kentookumura/exp301-gauge-edge-potential-train` version 2、id_no `128007163`。
- input manifest logical SHA: `8c213d886833a759ea193fcb0c5275e18187d37a9cb46e667663a6b86326f7ca`。
- identity logical SHA: `8fcf29d88693da05c2e36df13a96462a8ae0a3df8f07b46c69e19c1408f90d31`。
- support logical SHA: `fa6ed7a88ab4a348130a8d3dd0e26c41df64c50152272c1c64706f0a0925cae5`。
- Stage 1を実行していないためgrid solution / solver / OOF prediction / gzip SHAは未生成で対象外。
- model / submission SHA: 対象外。
- rerun result: 未実行。

## 解釈

6 formation edge identity自体は全foldで十分強く、eligible edge fractionも5/5 foldsで1.0だった。一方、250 ft / 4-neighbor /
halo 1の固定active gridは、outer-valid geometryを含めてもdonor constraintのないcomponentを残した。全query geometry
5,092,255 rows中90,827 rowsがdonor-supported component外で、厳密coverage 1.0を満たさない。既存prediction fallbackや
同一OOFでのgrid/halo/adjacency救済は事前contractで禁止しているため、Stage 1を開始しなかった。この結果はpotentialの
direct RMSEを否定するものではないが、現設計のfold-safe identifiabilityを否定するtechnical negative resultである。

## 次

本branchではinference、submission、案2/案3へ進まない。再訪する場合は別実験のgeometry-only component bridge readoutで、
unsupported componentと最近傍donor-supported componentの距離・連結安全性だけをtruth-freeに監査し、固定connectivity contractを
事前に定義できる場合に限る。
