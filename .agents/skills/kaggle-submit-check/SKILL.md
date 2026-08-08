---
name: kaggle-submit-check
description: "Kaggle 提出前に、提出物と notebook metadata を検証し、問題を切り分けて記録する。`submission.csv`、zip された予測ファイル、Kaggle Notebook output、`kernel-metadata.json`、`sample_submission` 互換性、hidden test 対応、行数、重複 ID、欠損、internet/GPU metadata、オフライン実行前提を確認する。Kaggle submit 直前、notebook output の検証、提出失敗時に使う。"
---

# Kaggle 提出チェック

ローカルで、アップロードしない事前チェックを実行する。このスキルでは、ユーザーが明示的に依頼しない限り、submit、push、upload を行わない。

## 手順

1. 対象を特定する。
   - CSV ファイル、zip ファイル、notebook フォルダ、またはプロジェクトルート。
   - 指定がなければ、`submission.csv`、`*.zip`、`kernel-metadata.json`、`sample_submission.csv` などの候補を探す。

2. 同梱 checker を実行する。

```bash
python .agents/skills/kaggle-submit-check/scripts/check_submission.py PATH --sample sample_submission.csv
```

`--sample` は sample file が存在する場合だけ使う。

3. スクリプトだけでは証明できない warning を手で確認する。
   - CV は test で想定される grouping/time split と一致しているか。
   - 出力は `sample_submission.csv` の行順を保っているか。
   - Kaggle のオフライン環境で依存関係を利用できるか。
   - ルールで許可されていない限り、internet が無効になっているか。
   - Kaggle Notebook の実行時間とメモリ制限に収まる推論になっているか。
   - code competition では公開 `test/` と `sample_submission.csv` が hidden test 用に差し替えられても動作するか。公開 test 固有の ID、行数、ファイル名、SHA、予測値に依存していないか。
   - hidden test の入力と保存済み model manifest / model 生成物だけで推論が完結し、実行時の sample submission と ID で 1 対 1 に整列できるか。

4. 結果を報告する。
   - `PASS`: チェックした範囲からは提出してよい。
   - `WARN`: ユーザーがリスクを受け入れる場合のみ提出してよい。
   - `FAIL`: 提出しない。正確な修正方法を示す。

## リポジトリ内の提出フロー

Kaggle 実験リポジトリ内で作業する場合:

1. `project.yml` に提出設定があることを確認する。sample file、id column、target columns を含む。
2. inference notebook の生成、push、Kaggle 実行は `kaggle-review-exp` と `kaggle-platform` に委譲する。このスキルでは、生成済み notebook と `kernel-metadata.json` を検証する。slug は 50 文字以内、`id` と `title` 由来 slug は一致、accelerator / internet / competition source は意図どおりであることを確認する。

3. `kaggle-platform` の手順で取得された Kaggle output がある場合、Kaggle 上で生成された `submission.csv` を sample submission に対して検証する。

```bash
task submit-check EXP=expXXX_title SUBMISSION=/tmp/kaggle-output/expXXX_title/inference/submission.csv
```

ローカル実験ディレクトリに提出 CSV を常設しない。Kaggle output を取得した場合だけ、その取得物に対して `task submit-check` を実行する。

4. PASS / WARN / FAIL を報告する。code competitionでは、Kaggle outputに存在する提出ファイル名（通常`submission.csv`）も引き渡し情報として明記する。実際の submit はユーザーが明示的に依頼した場合だけ `kaggle-platform` の手順で行い、`-k OWNER/SLUG -v VERSION -f submission.csv`の4点を省略しない。submit 後の監視は `kaggle-submit-monitor` に委譲する。

5. submit が行われてスコアが分かったら、リポジトリに記録する。

```bash
task record-submission EXP=expXXX EXTRA_ARGS="--cv 0.1234 --public-lb 0.1200 --notes baseline"
task update-summary
```

記録先:
- `SESSION_NOTES.md`: submit-check、提出コマンド、状態、次アクション。
- `result.md`: 提出結果の解釈、実行証拠、ユーザーの採否判断。日本語で記載する。
- `metrics.json`: CV/LBなど機械処理する数値の正。
- `experiment_summary.md`: 実験横断の要約。
- `submissions/SUBMISSIONS.md`: 提出履歴。

確認項目:
- 行数が `sample_submission.csv` と一致している。
- 必須列が存在し、想定外の追加列がある場合は意図的である。
- id column が設定されている場合、ID の順序と内容が一致している。
- missing、NaN、infinite values がない。
- offline/notebook competition では、ルールで許可されていない限り `enable_internet` が false。
- code competition では、公開 test 固有値に依存せず hidden test に差し替え可能である。
- 保存済み model manifest / model 生成物だけで推論でき、実行時の sample submission に ID で完全整列する。

## 提出物の種類

- CSV: header、行数、重複 ID、空値/NaN/Inf、sample 互換性を検証する。
- Zip: member、hidden file、nested path、含まれる CSV があればその内容を検証する。
- Kaggle Notebook: `kernel-metadata.json`、notebook の存在、source、internet/GPU flags、期待する output file を検証する。

このスキルは提出前検証に集中する。アップロード済みの提出を監視する場合だけ `kaggle-submit-monitor` を使う。
