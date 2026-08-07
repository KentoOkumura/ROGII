---
name: kaggle-strategy
description: "ローカルの実験メモ、`experiment_summary.md`、`KAGGLE_DIRECTION.md`、日次レポート、metrics、提出履歴、保存済みノートブック、保存済みディスカッション、既存の調査メモから Kaggle コンペ戦略を整理する。次の実験、次の一手、ロードマップ、週次戦略、CV/LB の一貫性、失敗パターン、CV 停滞、LB 変動、優先順位付け、実験横断の整理を求められたときに使う。新しい外部文献や過去解法の調査が必要なら、先に `kaggle-survey-papers` を使う。"
---

# Kaggle 戦略整理

単発の結果ではなく、複数実験から見える「流れ」を整理する。まずローカルの証拠を使う。新しい外部調査が必要な場合は、戦略を確定する前に `kaggle-survey-papers` を使うべきだと明示する。

## 手順

1. ローカル文脈を集める。

```bash
python .agents/skills/kaggle-strategy/scripts/collect_strategy_context.py --root .
```

2. 収集結果から、情報量の多いファイルを読む。
   - `docs/surveys/README.md`の実験番号・種類・トピック別索引と、そこから選んだ関連レポート
   - コンペ概要、または存在する場合は `KAGGLE_DIRECTION.md`
   - `experiment_summary.md`
   - `daily_reports/*`
   - `submissions/SUBMISSIONS.md`
   - `**/SESSION_NOTES.md`
   - legacyの`docs/experiment/*.md`、`docs/experiments/*.md`、または類似の実験ドキュメントが残る場合は、`docs/surveys/`に正がないか先に確認する
   - `**/metrics.json`

次の実験を提案するときは、少なくとも `experiment_summary.md`、`KAGGLE_DIRECTION.md`、`submissions/SUBMISSIONS.md`、最近の `experiments/*/SESSION_NOTES.md` を読む。

3. 作業を提案する前にフェーズを判定する。
   - Early: 安定したベースラインがなく、CV やデータ理解がまだ固まっていない。
   - Mid: ベースラインがあり、CV も信頼でき、特徴量やモデル探索が進んでいる。
   - Late: 締切が近い、または実験が頭打ち。アンサンブル、頑健性、提出管理を重視する。

4. 簡潔な戦略メモを出す。
   - 現在のフェーズと、フェーズ認識のずれ。
   - 現時点のベスト結果、信頼度、根拠ファイル。
   - CV/LB の一貫性評価。
   - 主な失敗パターンと、明確に効かなかったこと。
   - 次に試す手堅い実験。
   - 当たれば大きい高リスク実験。
   - もう少し長いロードマップが有用なら、期待値の高い次の 2-4 実験。
   - リスク管理: leakage、CV/LB 乖離、実行時間、提出回数制限。
   - `KAGGLE_DIRECTION.md` の「アイデアバックログ」に、完了済み・実装済みの候補が残っていないか。
   - 実験結果から出た次候補が backlog に反映され、既存候補も含めて優先度が見直されているか。

回答には次を含める。

- 現時点のベストスコアと、その信頼度。
- route 別の現時点の基準と、その根拠。
- CV/LB の一貫性評価。
- 主な失敗パターン。
- 次に試す手堅い実験。
- 当たれば大きい高リスクな実験。

## ルール

- すべての提案は、ローカルファイルのパス、Kaggle 公開情報、論文、または明示した仮定に結び付ける。
- 仮定は `Assumption:` としてラベルを付ける。
- 手法の中核機構を保った最小の反証可能実験を優先する。target、output表現、loss、decode、whole-group / local contextを変更して安くする場合は `proxy` であり、忠実実装より優先しない。
- ユーザーが特定手法またはrepresentation changeを求めた場合は、既存コードの再利用率、GPUコスト、実装容易性より手法忠実性を優先する。忠実実装のコストが大きい場合は、無断で縮小せず選択肢としてユーザーに示す。
- 同じ親実験または機構familyの `parameter tuning`、`add-only feature`、`selector-only`、後処理が2件連続した場合、またはpositiveなoracle headroom / coverage / 誤差非相関性に対しend-to-end改善が得られない場合は、次の実験を確定する前に `kaggle-idea-forge` でrepresentation auditを行う。
- 次実験の提案には、少なくとも1件の target / output / decode / context unit を変える高upside案を含める。採用しない場合は、safe案を選ぶ根拠を証拠とともに書く。
- フォルダ構成は作らない。ログやメモが不足している場合は、不足している証拠を明示しつつ、最小限の次アクションを示す。
- backlog を更新する場合は、実装済みアイデアを削除し、次候補の追加と同時に全候補の優先度を再評価する。
