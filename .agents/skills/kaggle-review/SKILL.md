---
name: kaggle-review
description: "Kaggle コンペのコードやノートブックをレビューし、失敗した実行を調査し、データや OOF 予測を分析する。leakage、CV/LB 乖離、実行時間制限、メモリ/OOM リスク、Kaggle Notebook のオフライン互換性、ROGII Wellbore Geology Prediction のワークフローを確認する。Kaggle コードレビュー、コンペコードレビュー、学習/推論パイプラインレビュー、ノートブックレビュー、失敗実行の調査、データ/OOF 分析、leakage/実行時間レビュー、metric/前処理監査を求められたときに使う。実験記録の監査には `kaggle-review-exp`、最終提出ファイルや kernel metadata の検証には `kaggle-submit-check` を使う。"
---

# Kaggle レビュー

コードレビューの姿勢で臨む。指摘は重要度順に並べ、可能ならファイルと行番号を付ける。スタイルよりも、正しさ、leakage、再現性、実行時間リスクを優先する。

## 対象範囲の決め方

- ユーザーがファイル、ディレクトリ、`staged`、git range を指定した場合は、その範囲をレビューする。
- 指定がなければ `git diff --staged` を確認し、空なら `git diff` を確認する。
- diff がなければ、プロジェクト内の提出/学習パイプラインらしいファイルをレビューする。

## レビュー優先度

1. Critical: データリーク、target/test 混入、不正な提出形式、非決定的なスコアリング、Kaggle で実行不能なコード。
2. High: CV/LB 乖離、誤った split/grouping、`/kaggle/input` のパス誤り、OOM/実行時間超過リスク、dtype/device バグ。
3. Medium: 壊れやすいエラー処理、seed 管理不足、非効率な推論、弱いログ、依存関係/バージョン問題。
4. Low: 保守性、命名、ローカル専用の前提。

## Kaggle 固有の確認

- train/validation/test の境界が守られているか確認する。
- ID と行順が `sample_submission.csv` と一致するか確認する。
- オフライン互換性を確認する。ルールで許可されていない限りインターネットを使わず、依存関係やデータは Kaggle input として用意されている必要がある。
- ファイルシステムの前提を確認する。ローカルパスは `/kaggle/input`、`/kaggle/working`、またはプロジェクト相対パスにきれいに対応するべき。
- メモリと時間を確認する。batch size、chunking、モデル読み込み、アンサンブル数、multiprocessing、cache の増加に注意する。
- 再現性を確認する。seed、決定的な fold、保存された config、正確なモデル生成物名を確認する。
- code competition では、公開 `test/` と `sample_submission.csv` が hidden test 用に差し替えられる前提で、入力列挙、ID 整列、メモリ、実行時間を確認する。公開 test 固有の ID、行数、ファイル名、SHA、予測値に依存する分岐を認めない。
- hidden test の入力と保存済み model manifest / model 生成物だけで推論が完結し、train-only 列、ローカル cache、公開 test の保存済み予測に依存しないことを確認する。

## 実験コードレビューの重点

実験コードや notebook をレビューするときは、次を優先する。

- データリークや fold 間の混入がないか。
- メトリック実装と最適化の向きに誤りがないか。
- 学習時と推論時で前処理がずれていないか。
- 学習済みモデル、前処理状態、特徴量名と順序、variant / mode / fold、相対パス、SHA が model manifest に保存され、推論側が manifest に基づいて読み込むか。
- 再現性を崩す未管理のランダム性がないか。
- Kaggle のオフライン環境で、実行時間やメモリに無理がないか。

## 失敗実行の調査

ユーザーが失敗した実行を報告した、または実験の debug を求めた場合:

1. 正確なコマンドと、最初に意味のある traceback を取得する。
2. 失敗を environment、data path、config、code、memory、network、Kaggle runtime に分類する。
3. コードを変える前に、`config.yaml`、`settings.py`、パス前提を確認する。
4. 最小の修正を入れ、失敗している最小範囲のコマンドを再実行する。
5. リポジトリのワークフローで実験メモが求められる場合は、再発する失敗を対象実験の `SESSION_NOTES.md` に記録する。

## データと OOF 分析

- まず schema、行数、欠損、重複、target 分布を確認する。
- 生の行を大量に出すより、要約表やプロットを優先する。
- `task oof-app` のようなプロジェクト内の確認アプリがあり、依存関係も入っている場合は活用する。`task` がない環境では同名の Make target を使う。
- 調査中の時系列記録は対象実験の`SESSION_NOTES.md`、通常の実験結果と証拠の解釈は`result.md`に保存する。独立した完了分析レポートを作る場合は、対象が単一実験でも実験横断でも`docs/surveys/`に保存する。
- 広い優先順位付けや次実験の統合判断には `kaggle-strategy` を使う。

## ROGII 固有の観点

ROGII Wellbore Geology Prediction では、特に次を確認する。

- well/group leakage が fold をまたいで発生していないか。
- depth/order を意識した validation になっているか。
- geology label と rare class の stratification。
- feature generation が validation/test の境界を覗いていないか。
- post-processing が class balance を不自然に変えたり、期待される提出 schema に違反したりしていないか。
- log/curve の単位整合性と欠損値処理。

## 出力形式

次の構成を使う。

```markdown
**Findings**
- [Severity] [file:line] 問題、影響、具体的な修正案。

**Open Questions**
- 正しさに実質的に影響する質問だけを書く。

**Test Gaps**
- 足りない確認、または実行していないコマンドを書く。
```

問題が見つからない場合は、その旨を明確に書き、それでも残るリスクを添える。
