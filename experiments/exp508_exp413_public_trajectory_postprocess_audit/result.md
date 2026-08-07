# exp508_exp413_public_trajectory_postprocess_audit 結果

## 結論

Kaggle private CPU Stage A version 1は完了した。固定SG61/p3は保存exp413 OOFを
`7.884802794404715 → 7.878669066831366`へ`0.00613372757334929 ft`改善し、5/5 folds、
固定5 scope、by-well tail、prediction-start continuity、technical gateをすべて通過した。
しかし事前固定したpooled gain `>=0.01 ft`だけを未達としたため、promotion gateはFAIL。
`FAIL_CLOSE_WITHOUT_SG_GRID_WARMUP_ROUTER_OR_GATE_RESCUE`として、推論・提出なしで終端閉鎖する。

## 実行

- Kernel: `kentookumura/exp508-exp413-public-sg61p3-audit-train`
- Version / id_no: `1 / 129625989`
- Runtime: private CPU、internet off、GPUなし
- 実行量: selectable primary 1、report-only 2、model / booster / HMM / PF / Beam / GPUすべて0
- 親・control再学習: 0
- 評価行 / well: `3,783,989 / 773`

## Primary結果

| メトリック | exp413 | SG61/p3 | delta / gain |
| --- | ---: | ---: | ---: |
| pooled RMSE | 7.884802794 | 7.878669067 | gain `+0.006133728 ft` |
| fold 0 | 7.919988324 | 7.914334781 | -0.005653543 |
| fold 1 | 8.377381333 | 8.369169936 | -0.008211397 |
| fold 2 | 7.539713352 | 7.534915364 | -0.004797988 |
| fold 3 | 7.574331167 | 7.568178439 | -0.006152728 |
| fold 4 | 7.982868393 | 7.977172131 | -0.005696261 |

全foldで改善した。固定scopeもすべて改善し、最小改善は`md_250_1000`の
`-0.004167012 ft`、最大改善は`hidden_like_spatial`の`-0.008780305 ft`だった。

## 安全性

- by-well delta p95: `-0.001344491 ft`
- worst-well delta: `-0.000417966 ft`（worst扱いでも改善）
- `+0.25 / +1 / +3 / +5 ft`悪化well数: `0 / 0 / 0 / 0`
- first score row補正量 p95 / max: `0.289606691 / 0.810694404 ft`
- second-difference RMS: `0.530397824 → 0.011204509 ft`
- technical / leakage / SHA checks: 全PASS

## Promotion gate

| 条件 | 判定 |
| --- | --- |
| technical all PASS | PASS |
| pooled gain `>=0.01 ft` | **FAIL**（`0.006133728 ft`） |
| nonworse folds `>=4/5` | PASS（`5/5`） |
| 固定5 scope delta `<=+0.02 ft` | PASS |
| by-well p95 / worst `<=+0.25 ft` | PASS |
| first-row p95 / max `<=0.50 / 2.00 ft` | PASS |

全ANDのため最終判定はFAIL。最小gainを結果後に緩和しない。

## Report-only

- `tau85_warmup_final_delta`: RMSE `7.886111093377577`
- `tau85_warmup_then_sg61_p3`: RMSE `7.880001331601432`

いずれもprimary decision freeze後にのみscoreし、`selectable=false`のまま救済・候補選択には
使っていない。

## 再現性

- primary prediction content SHA256: `5caf53f17c52729198eec412cdf7ce46e25f199a24ca842e4efb117e91a67f56`
- prediction parquet SHA256: `cf6135f20196ab319218b130ccf7682fb1c094684ded8f930b5c7079a1d53203`
- promotion gate SHA256: `d7e21a243e53f2d5af4e192faf3246e307f4ce92cb2f063bf5e0086582466d1b`
- reproducibility manifest SHA256: `1e68714488b1d7dd938dfe140ab67b00e7f93deb68efb6c520273afbe970b81a`
- deterministic anchor: `false`（同一環境rerunによるprediction SHA一致確認は未実施）

## 次

SG window/polyorder、tau、blend、reanchor、clip、projection、gateを同じOOFで調整しない。
exp508 PASSを前提にしたwell routerも作らず、inference / submissionへ進まない。独立した現行P1
候補を優先する。
