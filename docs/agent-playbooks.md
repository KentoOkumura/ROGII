# 作業別手順の参照先

この文書は人間向けの参照索引です。リポジトリ全体に常時適用する規約は [`AGENTS.md`](../AGENTS.md)、Codexの作業別手順の正は `.agents/skills/*/SKILL.md` です。この文書に作業手順を複製しません。

## 主な参照先

- 承認済み実験の作成、requirementsへの契約移行、Notebook実装、実行、記録、実験レビュー: [`kaggle-review-exp`](../.agents/skills/kaggle-review-exp/SKILL.md)。backlog経由では検証中の仮説とbacklog側の更新・削除を`kaggle-strategy`へ引き渡す。
- Kaggle API / CLI、Notebook push、ログ・output取得: [`kaggle-platform`](../.agents/skills/kaggle-platform/SKILL.md)
- コード、Notebook、失敗実行、OOFのレビュー: [`kaggle-review`](../.agents/skills/kaggle-review/SKILL.md)
- 提出ファイルとNotebook metadataの検証: [`kaggle-submit-check`](../.agents/skills/kaggle-submit-check/SKILL.md)
- 提出後の監視とLB記録: [`kaggle-submit-monitor`](../.agents/skills/kaggle-submit-monitor/SKILL.md)
- 次実験の候補生成: [`kaggle-idea-forge`](../.agents/skills/kaggle-idea-forge/SKILL.md)。検証中の仮説やbacklogへ直接保存しない。
- 検証中の上位仮説、実験戦略、優先順位、アイデアbacklogの作成・更新・削除: [`kaggle-strategy`](../.agents/skills/kaggle-strategy/SKILL.md)
- OOF誤差分析: [`kaggle-oof-readout`](../.agents/skills/kaggle-oof-readout/SKILL.md)。仮説・候補のbacklog反映は`kaggle-strategy`へ引き渡す。
- 論文、過去解法、公開Notebook、外部実装の調査: [`kaggle-survey-papers`](../.agents/skills/kaggle-survey-papers/SKILL.md)
- 公開Notebookの保存: [`kaggle-notebook-fetch`](../.agents/skills/kaggle-notebook-fetch/SKILL.md)
- Kaggle discussionの保存: [`kaggle-discussion-archive`](../.agents/skills/kaggle-discussion-archive/SKILL.md)
- Colab実行とGoogle Drive入出力: [`colab-notebook-runner`](../.agents/skills/colab-notebook-runner/SKILL.md)

## 共有仕様

- 検証中の上位仮説と未着手候補の索引: [`backlog/KAGGLE_DIRECTION.md`](../backlog/KAGGLE_DIRECTION.md)。候補詳細は[`backlog/`](../backlog/)。
- 再現性の確認: [`docs/06_reproducibility.md`](06_reproducibility.md)
- 完了した調査レポートの保存・検索・検証: [`docs/surveys/README.md`](surveys/README.md)
- 公式コンペ資料: [`docs/official/`](official/)
