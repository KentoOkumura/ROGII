# exp081_pilkwang_branch_decomposition 結果

## 仮説

Pilkwang final を branch 別に分解すると、full final を直接 submit するより前に、projected ridge/PF、pretrained LGBM、model-package tiny gate のどこが提出候補として価値を持つかを整理できる。

## 設定

- 親: exp079_public_artifact_replay_integrity_audit
- 検証: target-free branch decomposition
- メトリック: pairwise RMSE / MAE / max abs diff
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |
| Candidate count | 16 |
| Submit candidate count | 2 |
| Rank 1 | `submission_projected_ridge_pf_projection_d4_b075_raw.csv` |
| Rank 2 | `submission_projected_ridge_pf_pretrained_lgbm_w0.60.csv` |

## 再現性

- deterministic anchor: false
- seed policy: no_rng_used
- kernel version: 未実行。ローカル二次解析。
- feature content SHA: 対象外
- model SHA / manifest SHA: 対象外
- prediction SHA: exp079 submission summary の candidate SHA を再掲。
- submission SHA: exp079 submission summary の candidate SHA を再掲。
- rerun result: `decomposition_completed`

## 解釈

Pilkwang final は `submission_projected_ridge_pf_pretrained_lgbm_base.csv` と同一で、`w0.55` も pairwise では final と RMSE 0。model-package tiny gate は final から最大でも RMSE 0.019913 しか動かないため、単体の submit 価値は小さい。

ridge-sp に最も近い Pilkwang branch は raw projection (`submission_projected_ridge_pf_projection_d4_b075_raw.csv`) で、final との差は RMSE 1.442298、ridge-sp との差は RMSE 1.130190。2 番手は w0.60 blend で、final から RMSE 0.144364 と小さく、ridge-sp 差は 1.941010 まで縮む。

pretrained LGBM 単独は ridge-sp との差が大きく、model-package-only は final から大きく外れるため提出候補から外す。今回の output には候補 CSV 本体がないので、row-level guard と submit-check は次工程で必要。

## 次

1. 提出回数を使う場合は rank 1 / rank 2 の候補 CSV 本体を再取得して submit-check / row-level guard を行う。
2. 直接 submit を急がず、SP45 / fle3n / Koolbox 系の source slug 固定と追加監査に進む。
