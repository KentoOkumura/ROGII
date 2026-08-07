# exp442_symmetric_broad_jump_rate_transition_hmm 結果

## 状態

Kaggle private CPU Stage 0 version 1を完了し、`stage0_fail_closed`で終了した。
Stage 1、inference、submission、同一仮説の救済調整は行わない。

## 仮説

固定1%の対称broad rate branchが、exp209の通常時の安定性をほぼ維持しながら、
persistent under-response時だけ尤度で選べるmulti-bin escape pathを提供する。

## 実行

- kernel:
  `kentookumura/exp442-symmetric-broad-jump-rate-trans-hmm-train`
- version / id_no: `1` / `129101211`
- runtime: private CPU、internet disabled
- candidate: `jump_weight=0.01`、`broad_sigma_rate=0.02`
- Stage 0: 1候補×32 wells、156,088 rows、5 reporting folds
- parent control rerun / model / booster / PF / Beam / GPU: 全て0
- elapsed: `9,190.990 sec`
- peak RSS: `1.191 GiB`
- CV / Public LB / Private LB: なし / なし / なし

## Gate結果

| 区分 | 結果 |
| --- | --- |
| technical | 14 / 15 PASS |
| mechanism | 4 / 9 PASS |
| Stage 1 eligible | false |
| 最終判定 | `stage0_fail_closed` |

| 必須指標 | 実測 | gate | 判定 |
| --- | ---: | ---: | --- |
| full-773 runtime投影 | `222,019.844 sec` | `<=30,600 sec` | FAIL |
| broad branch responsibility | `0.00976695` | 監査値 | - |
| non-adjacent posterior edge mass | `0.00684557` | `>=0.001` | PASS |
| 将来rate方向一致 | `0.529732` | `>=0.60` | FAIL |
| 方向正fold | `5 / 5` | `>=4 / 5` | PASS |
| forward-cause episode SSE削減 | `0.2431%` | `>=10%` | FAIL |
| persistent episode SSE削減 | `-4.4385%` | `>=5%` | FAIL |
| persistent改善well | `9 / 16` | `>=10 / 16` | FAIL |
| persistent改善fold | `2 / 5` | `>=4 / 5` | FAIL |
| matched-control pooled RMSE delta | `-0.155414 ft` | `<=+0.02 ft` | PASS |
| matched-control by-well delta p95 | `+0.069364 ft` | `<=+0.25 ft` | PASS |

## 解釈

broad branchは約0.98%のresponsibilityと0.68%のnon-adjacent edge massを持ち、
「実際には使われなかった」失敗ではない。保存exp209 controlに対する集計値と
by-well p95も安全gate内だった。

一方、方向一致は全foldで0.5をわずかに上回るだけで、pooled `0.529732`に留まった。
forward-causeへの改善は0.24%しかなく、仮説対象のpersistent episodeは4.44%悪化した。
fold 3ではpersistent SSEが`-118.45%`と大きく悪化している。したがって、
低確率の対称multi-bin escape pathは作れたが、必要な方向と持続区間を選ぶ情報を
持たず、rate-lag対策として採用できない。

さらにfull OOF換算は約61.7時間で固定8.5時間gateを大幅に超える。
科学gateとruntime gateの双方が明確にFAILしているため、weight / sigma /
trigger / emission / grid / gateの調整、rerun、Stage 1は行わず閉じる。

## 再現性

- scientific contract SHA:
  `cd97572dc08d68e4a2018b27e0309cde3e129c3e35814953ec0f237048c60752`
- prediction logical SHA:
  `01fb64c820f4f68d8a0c8d7a8891f5ceedad7c8a12d647b79206fb0f2acfe59e`
- target-free diagnostic logical SHA:
  `cd6809a1bbaa07fa30f2989f5bf317a2d8be1fad8ec6f29b07a43b65b72330fd`
- transition diagnostic logical SHA:
  `2439708574644cc73f7f7647d04686a61566e67f3cf2726d144350c12c19bf5f`
- Kaggle `metrics.json` SHA:
  `92d09d57c9afc0ff7397a54d231424ddfbe172478c1e91381ddab946e3ed53c7`

初回成功runだけなのでdeterministic anchorとは扱わない。ただし、全AND gateを
FAILした候補を再評価するための独立rerunは実施しない。

## 次

exp442は完了済みとしてbacklogから削除する。exp441/442/443の失敗をまとめると、
単にrate supportや格子平均を広げるだけではpersistent lagを安全に回復できない。
既存のexp444/exp446のように明示的なtrend/persistenceを扱う仮説は独立候補として
のみ扱い、exp442のFAILをpositive evidenceやgate救済には使わない。
