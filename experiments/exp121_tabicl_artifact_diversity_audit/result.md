# exp121_tabicl_artifact_diversity_audit 結果

## 仮説

TabICL / 保存済み artifact-stack 予測は、単体本命ではなく後続のアンサンブル候補として、既存 anchor と低相関な多様性材料に使える可能性がある。まずは保存済み CSV の互換性、SHA、pairwise / by-well distance を確認する。

## 設定

- 親: `tabicl_artifact_diversity_audit` backlog
- 検証: `target_free_submission_diversity_audit`
- メトリック: pairwise RMSE / MAE / p95 abs / max abs
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |
| 実行状態 | Kaggle train v3 完了 |
| Valid submissions | 7 |
| Candidate/reference submissions | 3 |
| Anchors | 4 |
| Pairwise rows | 15 |

Kaggle train v3 は `kentookumura/exp121-tabicl-artifact-diversity-audit-train` version 3。output は `/tmp/kaggle-output/exp121_tabicl_artifact_diversity_audit/train_v3`。

## Candidate vs Anchor 距離

| Candidate | closest anchor | RMSE | p95 abs | max abs |
| --- | --- | ---: | ---: | ---: |
| `needless_sel15_tabicl_public_output__002__submission` | `anchor_exp082_fle3n_final_source_port` | 1.220332 | 2.598166 | 4.135641 |
| `kojimar_pf_beam_tabicl_stack_output__001__submission` | `anchor_exp082_fle3n_final_source_port` | 1.447558 | 2.737982 | 4.211407 |
| `thbdh_v10_fresh_artifact_infer_output__000__submission` | `anchor_exp063_old_ml_public_replay` | 1.809928 | 3.703239 | 6.241670 |

主要 anchor との差分:

| Candidate | vs exp027 | vs exp063 | vs exp073 | vs exp082 |
| --- | ---: | ---: | ---: | ---: |
| `kojimar_pf_beam_tabicl_stack_output__001__submission` | 2.970821 | 2.660961 | 2.629900 | 1.447558 |
| `needless_sel15_tabicl_public_output__002__submission` | 2.701692 | 2.860030 | 2.650150 | 1.220332 |
| `thbdh_v10_fresh_artifact_infer_output__000__submission` | 5.621935 | 1.809928 | 2.045449 | 3.679163 |

Kaggle v1/v2 では TabICL candidate 3 件は取得できたが、anchor kernel outputs が expected path に mount されず anchor_count 0 だった。v3 では anchor 4 件を bootstrap input として同梱し、candidate-vs-anchor を完了した。

## 再現性

- deterministic anchor: false
- seed policy: `no_rng_used`
- kernel version: `kentookumura/exp121-tabicl-artifact-diversity-audit-train` v3
- feature content SHA: feature 生成なし
- model SHA / manifest SHA: モデルなし
- prediction SHA: Kaggle v3 の `artifacts/tabicl_artifact_diversity_audit_inventory.csv` に保存済み
- submission SHA: 提出候補は生成しない
- rerun result: v3 `audit_completed`

## 解釈

3 件の TabICL / artifact-stack candidate はいずれも既存 anchor と完全一致ではない。`needless` と `kojimar` は exp082 final source-port に近く、RMSE 1.2-1.45 程度の近傍候補。`thbdh v10 fresh artifact` は exp063/exp073 に近く、exp027/exp082 とは大きめに離れる。OOF は無いため誤差相関や改善判断はできず、単体提出候補ではなく、後続のアンサンブル候補を絞るための diversity inventory として閉じるのが妥当。

## 次

TabICL 系を使う場合は、直接 submit ではなく exp082 / exp092 系のアンサンブル候補 / candidate diversity source として、OOF がある候補だけ error correlation を確認する。
