# 実験ワークフロー

リポジトリ全体の規約は `AGENTS.md` を正とします。この文書は実験の流れだけを示し、Kaggle CLI や notebook の操作手順は `.agents/skills/` に集約します。

## 実験の流れ

1. `kaggle-strategy` が、ユーザーの承認した未着手候補を `KAGGLE_DIRECTION.md` と `docs/backlog/<candidate>.md` に記録する。
2. 実験開始時に`kaggle-review-exp`が候補詳細の契約、根拠、判断履歴をsteeringの`requirements.md`へ移し、`design.md`に実装対応、`tasklist.md`に作業順序を記録して実験ディレクトリを作る。移行確認後、`kaggle-strategy`が元のbacklog項目を削除する。
3. 同じ `EXP=expXXX_name` の中で train と inference を実装し、Kaggle Notebook のフル実行で評価する。
4. 実行証拠と比較結果を`AGENTS.md`の記録責務に従って一度ずつ記録し、完了・採用・不採用はユーザーの判断後に確定する。

作成時の入口だけを以下に示します。

```bash
task validate-config
task new-steering EXP=expXXX_name
task new-exp EXP=expXXX_name SOURCE=experiments/<parent-exp>
task validate-exp EXP=expXXX_name EXTRA_ARGS="--allow-todo"
```

実験固有テストは `experiments/<exp>/tests/` に置きます。親実験のテストは既定ではコピーせず、同じ契約を意図的に引き継ぐ場合だけ `--copy-tests` を使って期待値と参照先を見直します。複数実験またはリポジトリ全体の契約だけをルートの `tests/` に置きます。

## 作業別の参照先

- 実験作成、notebook 実装、Kaggle train/inference、実験記録、レビュー: `kaggle-review-exp`
- Kaggle CLI、認証、kernel push/pull/logs/output、slug、data sync: `kaggle-platform`
- 提出物と notebook metadata の検証: `kaggle-submit-check`
- submit 後の scoring 監視と LB 記録: `kaggle-submit-monitor`
- 実験横断の優先順位と backlog: `kaggle-strategy`

全 skill の入口は `docs/agent-playbooks.md` を参照してください。

## Kaggle 実行と output

Kaggle Notebook のフル実行と公式評価を正とします。local smoke は必要な入力と依存関係がローカルに揃う場合だけ使い、その結果だけで公式スコアや実行完了を判断しません。

CV は live logs、notebook cell、Kaggle UI の表示から記録できます。`submission.csv`、OOF、model manifest、feature importance、SHA、後続実験の入力など、実ファイルを確認する必要がある場合だけ Kaggle output を取得します。取得した提出物を検証するときだけ、明示したローカルパスに対して `kaggle-submit-check` を実行します。

Notebook-only code submission では、提出前に対象 kernel version の output に `submission.csv` が存在することを確認し、`task submit-code` を一度だけ実行します。同じ提出に raw CLI と task の両方を実行しません。正確な引数と失敗時の切り分けは `kaggle-platform` を参照します。

push 前に Kaggle UI の Active Sessions 確認は行いません。生成済み metadata の `enable_tpu: false` を確認し、GPU を使う場合だけ CLI で quota を確認します。TPU はこのリポジトリでは未対応です。同時 session 上限エラーが返った場合にだけ、待機または停止対象をユーザーへ確認します。

## 記録と判断

実験記録の各ファイルの役割、実験status、ユーザー判断の規則は`AGENTS.md`を正とします。このworkflowでは、スコア確定時に`record-exp`を先に実行し、その後`record-submission`で同じ値を再入力せず提出履歴を更新します。

再現性の証拠と rerun 方針は `docs/06_reproducibility.md` を参照します。実験を越える完了済み調査は `docs/surveys/` に保存し、`docs/surveys/README.md` を検索入口にします。
