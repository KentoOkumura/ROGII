# exp339 結果

## 状態

Kaggle CPU Stage 0 version 1完了。固定AND gateをFAILし、exp341を解禁せず枝を閉じた。HMM、TVT予測、inference、submission、LBはない。

## 結果

| fold | wells / coverage | rows | primary NLL | global NLL | circular NLL | real < circular | variance / MSE | length-sigma Spearman |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |
| 0 | 155 / 100% | 22,168 | 3.976947 | 4.001586 | 4.011631 | PASS | 1.151788 | 0.454771 |
| 1 | 155 / 100% | 21,678 | 3.989183 | 4.028110 | 4.059772 | PASS | 1.029019 | 0.313276 |
| 2 | 155 / 100% | 25,088 | 4.196907 | 4.243896 | 4.142608 | FAIL | 0.740724 | 0.521963 |
| 3 | 154 / 100% | 22,341 | 4.021002 | 4.055222 | 3.996664 | FAIL | 1.058740 | 0.672744 |
| 4 | 154 / 100% | 25,183 | 4.005850 | 4.033316 | 4.005378 | FAIL | 1.045877 | 0.606969 |

- pooled 116,458 rows / 773 wells。primary NLL `4.041311`はglobal constant `4.075874`より良く、5/5 foldsで改善した。
- pooled variance/MSE比は`0.975582`、fold別も5/5が固定範囲内。gap長と推定sigmaのSpearmanはpooled `0.518407`、5/5 foldsで正だった。
- pooledではprimary NLLがcircular `4.044584`より良いが、fold別勝利は2/5で、事前条件4/5を満たさなかった。他の10 checksはすべてPASSした。

## 解釈

gap長・anchor距離に応じた表はglobal分散より一貫してNLLを改善し、平均校正も良い。しかし実欠損run分布へ合わせたplacement固有の優位性はfold間で再現せず、固定表を自然欠損へ転送する根拠が不足した。事前規約どおりbin、support、pseudo-gap数、補間法を結果後に調整せず、exp341は依存FAILで閉じる。

## 次

同じmissing-gap tableの救済案は追加しない。既存の独立した0-booster候補exp340（depth-alias confidence readout）を次のP1--P2候補として維持し、robust emissionのexp342とACF temperingのexp343は低-中P3のまま後置する。

## 再現性

- Kaggle kernel: `kentookumura/exp339-missing-gap-pseudomask-uncertainty-train` version 1、id_no `128226213`
- runtime: `320.79614 sec`、CPU / internet off
- scientific contract content SHA: `798a4feeefbeb6c50390cc112e87c9ce07f257ac7c972dd25e9d73f73a43b9e0`
- fold assignment decompressed SHA: `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`
- pseudo-gap plan content SHA: `f77f379ad5f07ef8b2d792fe91320f059d98e3244893453c50b67673f3db9cbe`
- interpolation prediction content SHA: `d94efc3357b4ba4c3921d9d42db019ccff686dd5e1fe797437ce6979d69d656b`
- uncertainty table content SHA: `6a9bd955a64ab60fc442e6675c2676828f5426a74fdae3e6fbbd68e34d0eb4e5`（FAILのためexp341へ渡さない）
- fold summary decompressed SHA: `f46d8d35c502d5db644e6cdd2ecd407a274e86c09fb04d69929393a92c90d594`
- summary raw SHA: `b626969cb2b4452b9c8b8dd104c2c8337bfab619bb8cb7b2e782dd3c96e43fbd`
- Kaggle logsに加え、`metrics.json`、fold summary、summary、scientific contractの小さい4ファイルだけを対象取得した。大きいoutput archiveは取得していない。
