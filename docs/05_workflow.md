# 実験ワークフロー

リポジトリ全体の規約は `AGENTS.md` を正とします。この文書は実験の流れだけを示し、Kaggle CLI や notebook の操作手順は `.agents/skills/` に集約します。

## 実験の流れ

1. 実験化の入口を確認する。既存backlog候補を実験化する場合は、`docs/backlog/<candidate>.md`とそこから直接参照される根拠を引き継ぐ。ユーザーが直接実験化を承認した場合は、形式的なbacklogを作らない。
2. `kaggle-review-exp`が`experiments/<exp>/`を作り、同じディレクトリの`requirements.md`に契約、根拠、実装方法、受け入れ条件をまとめる。作業順序と進捗は`SESSION_NOTES.md`へ記録する。backlog経由では上位仮説ID、候補名、判断履歴を`requirements.md`と`config.yaml`へ移し、移行確認後に`kaggle-strategy`が元のbacklog項目を削除して検証中の仮説を更新する。直接実験化では、明示的に紐づける既存仮説がなければ`lineage.hypothesis_id`と`lineage.backlog_candidate`を`N/A`とする。
3. 同じ `EXP=expXXX_name` の中で、実験契約に必要な train、inference、audit、diagnostic のnotebookだけを実装し、必要なnotebookをKaggleでフル実行して評価する。雛形にtrainとinferenceの両方があっても、学習を伴わない監査や提出を目的としない実験へ不要な実装・pushを追加しない。
4. 実行証拠と比較結果を`AGENTS.md`の記録責務に従って一度ずつ記録し、完了・採用・不採用はユーザーの判断後に確定する。
5. 上位仮説を閉じる場合は関連実験を横断してユーザー判断を得て、上位仮説IDを持つ結論を`docs/surveys/`へ保存し、上位仮説別索引への反映後に`kaggle-strategy`が検証中の仮説から外す。

作成時の入口だけを以下に示します。

```bash
task validate-config
task new-exp EXP=expXXX_name SOURCE=experiments/<parent-exp>
task validate-exp EXP=expXXX_name EXTRA_ARGS="--allow-todo"
```

実験固有テストは `experiments/<exp>/tests/` に置きます。親実験のテストは既定ではコピーせず、同じ契約を意図的に引き継ぐ場合だけ `--copy-tests` を使って期待値と参照先を見直します。複数実験またはリポジトリ全体の契約だけをルートの `tests/` に置きます。

## 作業別の参照先

- 実験作成、実験契約に必要な Kaggle Notebook の実装・実行、実験記録、レビュー: `kaggle-review-exp`
- Kaggle CLI、認証、kernel push/pull/logs/output、slug、data sync: `kaggle-platform`
- 提出物と notebook metadata の検証: `kaggle-submit-check`
- submit 後の scoring 監視と LB 記録: `kaggle-submit-monitor`
- 検証中の上位仮説、実験横断の優先順位と backlog: `kaggle-strategy`

全 skill の入口は `docs/agent-playbooks.md` を参照してください。

## Kaggle 実行と output

Kaggle Notebookの公式評価とlocal smokeの扱い、およびsubmissionの承認条件は`AGENTS.md`を正とします。Kaggle outputを取得する条件、Notebook-only code submission、push前のruntime resource / quota確認は[`kaggle-platform`](../.agents/skills/kaggle-platform/SKILL.md)を正とします。取得済みの提出物とnotebook metadataの検証は`kaggle-submit-check`を使います。

## 記録と判断

実験記録の各ファイルの役割、実験status、ユーザー判断の規則は`AGENTS.md`を正とします。このworkflowでは、スコア確定時に`record-exp`を先に実行し、その後`record-submission`で同じ値を再入力せず提出履歴を更新します。

再現性の証拠と rerun 方針は `docs/06_reproducibility.md` を参照します。実験前の文献・公開Notebook調査、単一実験の完了分析、実験横断の完了調査は `docs/surveys/` に保存し、`docs/surveys/README.md` を検索入口にします。
