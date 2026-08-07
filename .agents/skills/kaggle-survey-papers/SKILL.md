---
name: kaggle-survey-papers
description: "Kaggle コンペに関係する論文、過去の Kaggle 解法、公開ノートブック、ディスカッション、ベンチマーク、GitHub 実装、ドメイン手法を調べる。論文調査、関連研究、過去解法、手法探索、転用できる実験案の作成を求められたときに使う。"
---

# Kaggle 関連研究調査

このスキルでは最新の Web 調査が必要になる。一次情報または質の高い情報源を確認し、短く実行可能な調査メモにまとめる。

## 手順

1. 検索クエリを決める。
   - ユーザーがキーワードを指定していれば、それを使う。
   - 指定がなければ、コンペ名、タスク、metric、データ形式、target から推定する。

2. 次の範囲を調べる。
   - Kaggle のコンペディスカッションと公開ノートブック。
   - 必要に応じて arXiv、PubMed、会議論文。
   - Papers With Code または公式ベンチマークページ。
   - 論文や既知の解法と結び付いている場合のみ、GitHub 実装。

3. 有用な情報源ごとに記録する。
   - タイトル、URL、年または日付
   - タスクと metric
   - モデル、特徴量、validation のアイデア
   - このコンペに転用できそうな理由
   - 実装難度
   - leakage、実行時間、オフライン実行のリスク

4. 次のテンプレートを参考に、Markdown メモとして保存するか保存案を示す。

```text
.agents/skills/kaggle-survey-papers/references/survey-template.md
```

このリポジトリでは、完了した外部調査レポートの正を`docs/surveys/`に置く。まず`docs/surveys/README.md`をトピック別に検索し、同じ問いなら既存レポートを更新する。新しい問いなら次のコマンドでYAML front matter付きレポートを作成する。

```bash
task new-survey-report SURVEY_TITLE="..." SURVEY_SLUG="..." EXTRA_ARGS="--type survey --type literature_review --topic ..."
```

論文単位の読書メモは`docs/papers/`、Kaggle discussionのアーカイブは`docs/discussions/`に残し、統合した結論をsurveyレポートから参照する。完了時は`status: final`と一行`summary`を設定し、`task update-survey-index`と`task validate-surveys`を実行する。

## 出力

文献リストから始めず、最も実行価値の高い 3-5 個のアイデアを先に出す。使った情報源へのリンクを含める。事実と、転用可能性に関する仮説を分けて書く。

Kaggle の公開ノートブックやディスカッションから、その手法を無効化するコンペ固有の制約が見つかった場合は、論文を過大評価しない。
