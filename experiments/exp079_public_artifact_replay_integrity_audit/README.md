# exp079_public_artifact_replay_integrity_audit

## 状態

- ルート: pf_beam
- 状態: audit_completed
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-18
- 親実験: public notebook route backlog

## 仮説

公開 notebook route の候補は、公開 LB や title だけでは採用できない。外部生成物依存、static visible CSV、hidden rerun 互換、branch output の中身、既存 anchor との差分を先に監査すれば、直接 submit する前に replay 候補としての信頼度を切り分けられる。

## 変更点

- Pilkwang notebook を最初の監査対象にした。
- ridge-sp / fle3n / SP45 / Koolbox 系を同じ監査枠で扱える `source_specs` を追加した。
- 外部 input / kernel output の inventory と SHA を保存する audit module を追加した。
- 候補 `submission.csv` / branch CSV の sample 互換性、予測範囲、SHA、pairwise distance を保存する。
- train / inference notebook は no-submit audit とし、提出ファイルを作らない。

## 検証方針

- Fold: なし
- Group: なし
- Stratification: なし
- Leakage Check: static visible CSV、public sample branch、hardcoded input submission、exact/override pattern を notebook source と artifact inventory から確認する。

## 実行入口

- 学習 notebook: `exp079_public_artifact_replay_integrity_audit_train.ipynb`
- 推論 notebook: `exp079_public_artifact_replay_integrity_audit_inference.ipynb`
- Kaggle 準備: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp079_public_artifact_replay_integrity_audit --notebook train --run-on-push --strict`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |
| Audit status | audit_completed |
| Kaggle kernel | kentookumura/exp079-public-artifact-audit-train v4 |
| Candidate files | 28 |
| Valid submission CSVs | 17 |
| Pairwise distances | 136 |

## 所見

### 良かった点

- 公開 notebook を提出候補にする前の確認項目を機械的に保存できるようにした。
- gzip 生成物は raw SHA に加えて decompressed content SHA を保存する。

### 悪かった点

- v1 は長い canonical slug で `SaveKernel` 400、v2 は Kaggle mount path 解決不足で `blocked_missing_required_sources`、v3/v4 で解消した。
- fle3n / SP45 / Koolbox 系は exact slug 未確定のため placeholder 扱い。

### リスク / 注意

- この実験は deterministic anchor ではない。
- 監査は通ったが、直接 submit はまだしない。まず branch decomposition と candidate selection を行う。
- 公開 notebook の output source に `.ipynb` が含まれない場合、source-level risk pattern は別途 notebook pull で補う必要がある。

## 次

- `pilkwang_branch_decomposition` として、final / projected ridge-PF / pretrained LGBM / model package only / gated candidates の寄与を整理する。
- fle3n / SP45 / Koolbox 系の exact source slug を固定して追加監査する。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
