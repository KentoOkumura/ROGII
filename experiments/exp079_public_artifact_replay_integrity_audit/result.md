# exp079_public_artifact_replay_integrity_audit 結果

## 仮説

公開 notebook route の候補は、外部生成物と static CSV リスクを監査してからでないと replay / submit 候補として扱えない。

## 設定

- 親: public notebook route backlog
- 検証: target-free integrity audit
- メトリック: integrity audit counts / pairwise submission distance
- シード: no_rng_used
- Kaggle kernel: `kentookumura/exp079-public-artifact-audit-train`
- 正の実行: version 4
- output: `/tmp/kaggle-output/exp079_public_artifact_replay_integrity_audit/train_v4`

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |
| Audit status | audit_completed |
| Missing required sources | 0 |
| Notebook inspections | 2 |
| Candidate files | 28 |
| Valid submission CSVs | 17 |
| Pairwise distances | 136 |

## 主な確認結果

- Pilkwang と ridge-sp の Kaggle sources は `/kaggle/input/notebooks/...` と `/kaggle/input/datasets/...` に mount され、version 4 で監査可能だった。
- Pilkwang final `submission.csv` は `submission_projected_ridge_pf_pretrained_lgbm_base.csv` / `submission_projected_ridge_pf_pretrained_lgbm_w0.55.csv` と完全一致した。
- Pilkwang final SHA: `53986e96fdf30b311a3298bea51849e5ec5088aa007f8d36ceab001ffa76e07f`
- Pilkwang final と projected ridge/PF projection の pairwise RMSE は 1.299277767。
- Pilkwang final と pretrained LGBM branch の pairwise RMSE は 1.588006160。
- Pilkwang final と model-package-only の pairwise RMSE は 17.318521442。
- Pilkwang final と ridge-sp final の pairwise RMSE は 2.020019968。
- model package gated candidates は base から tiny diff で、gmax 0.003 / 0.005 / 0.010 の RMSE はそれぞれ 0.005973822 / 0.009956370 / 0.019912740。
- Pilkwang notebook の source risk hits は `exact_match_or_override=38`、`mentions_sample_submission=26`、`writes_submission_csv=3`。ただし config 上の exact-match recovery / guarded overlap override は無効という前提は維持する。
- ridge-sp notebook は `writes_submission_csv=1`、`reads_submission_csv=2`、`mentions_public_or_visible=1`。

## 再現性

- deterministic anchor: false
- seed policy: no_rng_used
- kernel version: 4
- feature content SHA: 対象外
- model SHA / manifest SHA: 対象外
- prediction SHA: candidate CSV SHA を `exp079_public_artifact_replay_integrity_audit_submission_summary.csv` に保存
- submission SHA: candidate CSV SHA を `exp079_public_artifact_replay_integrity_audit_submission_summary.csv` に保存
- rerun result: v3 / v4 とも `audit_completed`。v4 は pairwise label を `source_name::label` に一意化した正の結果。

## 解釈

Pilkwang / ridge-sp の主要 output は code competition runtime 上で mount され、sample ID 互換性、予測範囲、branch-level pairwise distance を監査できた。公開 notebook route の primary audit は完了。ただしこれは submit 許可ではなく、次は Pilkwang branch decomposition と候補選定を行う段階。

fle3n / SP45 / Koolbox 系は exact slug 未固定のため、今回の audit では placeholder のまま。別途 source slug を固定して追加監査する。

## 次

1. `pilkwang_branch_decomposition` を実行し、final / projected ridge-PF / pretrained LGBM / model-package-only / tiny gate の寄与を整理する。
2. 提出するなら full final からではなく、branch diff と risk guard を読んで 1-2 candidate に絞る。
3. fle3n / SP45 / Koolbox 系の exact source slug を固定して追加監査する。
