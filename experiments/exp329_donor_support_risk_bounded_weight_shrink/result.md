# exp329 結果

## 状態

Kaggle CPU Stage 0完了・scientific gate FAIL・branch closed。Stage 1、inference、submissionへ進めない。

canonical kernelは`kentookumura/exp329-donor-support-risk-shrink-train`。version 1はNumPy 2のobject列比較で科学判定前に停止し、dtype-aware identity guardだけを修正したversion 2が`COMPLETE`になった。

## Stage 0判定

| 指標 | 結果 | 事前gate | 判定 |
| --- | ---: | ---: | --- |
| pooled real AUC | 0.562091 | 0.60以上 | FAIL |
| AUC > 0.5 folds | 5/5 | 4/5以上 | PASS |
| pooled control AUC | 0.556781 | 診断値 | - |
| real - control AUC | 0.005310 | 0.05以上 | FAIL |
| top-risk mean benefit | -0.674259 ft | 0.10 ft以上 | FAIL |
| bottom-risk mean benefit | -1.343711 ft | 診断値 | - |
| top - bottom benefit | 0.669452 ft | 0.25 ft以上 | PASS |
| 1000+方向非悪化 | false | true | FAIL |
| hidden-like spatial方向非悪化 | false | true | FAIL |
| hidden-like typewell-purged方向非悪化 | false | true | FAIL |

発火は762,529 / 3,783,989行（20.1515%）、433 wells、5 foldsで、事前coverage gateは全PASSした。technical hard checksもexp263式parity、fold、row/well identity、finite coverage、K16 segment coverageを含め全PASSした。したがって失敗理由はcoverageや実装ではなく、donor-support riskの科学的識別力と補正方向の不足である。

## 再現性

- 773 wells / 12,368 segments / 3,783,989 rows。
- Stage 0本体runtime: 209.829秒。
- target-free contract SHA256: `03049211fdf9c394ff7c34426e0cbb0ab424da3ae440ab92136c106b805f3000`。
- donor ledger decompressed SHA256: `99c668986af8e5857f772be513e9a89a95e20a216db6a39425cf2cdbf258053e`。
- support primitives decompressed SHA256: `1c03ac22d39b20380a42b392d65017268633390b2d1371b895fba8977ad15bd2`。
- segment risk decompressed SHA256: `9392018315617b0974162c7edbb20b7b75602572b2d3e0c70b4ae3af77de71b5`。
- Stage 0 decision raw SHA256: `9c25c68c12527002fd1171dd5dea39448f8eb1928d495a88b48513c65cf0f8a2`。

## 次

事前登録どおり、threshold、alpha、clip、destination、featureの救済gridを行わずbranchを閉じる。exp329 Stage 0 PASSを必須依存にしていたexp330も未実装・未実行のまま閉じ、新しい同系救済backlogは追加しない。
