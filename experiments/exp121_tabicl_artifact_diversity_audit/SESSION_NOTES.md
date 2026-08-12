# exp121_tabicl_artifact_diversity_audit セッションノート

## 目的

TabICL / 保存済み artifact-stack 予測を、単体提出ではなく後続のアンサンブル候補として既存 anchor との多様性材料に使えるか target-free に監査する。TabICL 再推論、GPU、モデル学習、提出候補生成は行わない。

## 現在の状態

- Route: ensemble
- 状態: Kaggle train v3 完了
- CV: まだなし
- LB: まだなし

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 2026-06-25

```bash
make new-steering EXP=exp121_tabicl_artifact_diversity_audit
make new-exp EXP=exp121_tabicl_artifact_diversity_audit
```

- `docs/legacy/steering/20260625-exp121-tabicl-artifact-diversity-audit/` と `experiments/exp121_tabicl_artifact_diversity_audit/` を作成。
- CPU-only 監査として実装する方針を確定。GPU は不要。

### Kaggle train 実行

```bash
make prepare-kaggle-notebooks EXP=exp121_tabicl_artifact_diversity_audit EXTRA_ARGS="--notebook train --run-on-push --strict"
kaggle kernels push -p experiments/exp121_tabicl_artifact_diversity_audit/kaggle/train
```

- 初回 push は title slug mismatch で 400。`--title 'exp121 tabicl artifact diversity audit train'` と明示して解消。
- v1: `kentookumura/exp121-tabicl-artifact-diversity-audit-train` version 1 complete。TabICL / artifact candidate 3 件は取得できたが、anchor は local `/tmp` / repo path 依存で missing、`audit_completed_no_anchors`。
- v2: anchor kernel sources を metadata に追加したが、runtime の expected path に anchor outputs が展開されず、同じく `audit_completed_no_anchors`。
- v3: anchor 4 件を `anchor_inputs/*.csv` として bootstrap に同梱し、`audit_completed`。output: `/tmp/kaggle-output/exp121_tabicl_artifact_diversity_audit/train_v3`。
- v3 summary:
  - valid submissions: 7
  - candidate submissions: 3
  - anchor submissions: 4
  - pairwise rows: 15
  - by-well pairwise rows: 45
  - GPU required: false
  - TabICL rerun performed: false
  - submission candidate created: false

## 変更点

- `config.yaml` に `ensemble` route、CPU-only Kaggle metadata、候補 source root、比較 anchor を記載。
- `tabicl_artifact_diversity_audit.py` を追加。candidate / anchor CSV の inventory、pairwise distance、by-well distance、summary、README、metrics を保存する。
- source が無い場合は missing として記録し、監査自体は正常終了する。
- local smoke:
  - command: `.venv/bin/python experiments/exp121_tabicl_artifact_diversity_audit/tabicl_artifact_diversity_audit.py`
  - status: `audit_completed`
  - valid submissions: 20
  - candidate/reference submissions: 16
  - anchors: 4
  - pairwise rows: 184
  - TabICL 固有 source は local では missing。exp082 fle3n source-check 由来 reference は読み込めた。
- validation:
  - `.venv/bin/python -m py_compile ...` PASS
  - notebook JSON check PASS
  - `make validate-exp EXP=exp121_tabicl_artifact_diversity_audit` PASS
  - `make prepare-kaggle-notebooks EXP=exp121_tabicl_artifact_diversity_audit EXTRA_ARGS="--notebook train --strict"` PASS
  - generated metadata: GPU false, internet false, dataset sources 4 件, kernel sources 3 件
- v3 で `runtime.kaggle.bootstrap_files` に anchor CSV 4 件を追加し、Kaggle runtime 上で exp027 / exp063 / exp073 / exp082 anchors を比較できるようにした。

## v3 主要距離

| Candidate | closest anchor | RMSE | p95 abs | max abs |
| --- | --- | ---: | ---: | ---: |
| `needless_sel15_tabicl_public_output__002__submission` | `anchor_exp082_fle3n_final_source_port` | 1.220332 | 2.598166 | 4.135641 |
| `kojimar_pf_beam_tabicl_stack_output__001__submission` | `anchor_exp082_fle3n_final_source_port` | 1.447558 | 2.737982 | 4.211407 |
| `thbdh_v10_fresh_artifact_infer_output__000__submission` | `anchor_exp063_old_ml_public_replay` | 1.809928 | 3.703239 | 6.241670 |

## 再現性メモ

- seed policy: `no_rng_used`
- stochastic components: なし
- CPU/GPU runtime: CPU-only。`runtime.kaggle.enable_gpu=false`
- Kaggle kernel id / version: `kentookumura/exp121-tabicl-artifact-diversity-audit-train` v3
- input / feature schema SHA: feature schema なし。candidate / anchor CSV の SHA を inventory に記録する。
- feature content SHA: feature 生成なし
- model manifest / model SHA: モデルなし
- prediction SHA: candidate / anchor CSV の SHA を v3 `artifacts/tabicl_artifact_diversity_audit_inventory.csv` に保存済み。
- submission SHA: 提出候補は生成しない
- rerun check: v3 complete

## 次のアクション

1. 直接 submit はしない。
2. TabICL 系を使うなら、後続のアンサンブル候補として扱い、OOF がある候補だけ error correlation を確認する。
