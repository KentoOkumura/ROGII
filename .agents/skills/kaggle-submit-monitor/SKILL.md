---
name: kaggle-submit-monitor
description: "`kaggle competitions submit` 後の Kaggle 提出を監視する。特に scoring が長い code competition 向け。提出の監視、実行時間の計測、LB/public score の記録、submission log の tail、`exp002_fold0` のような実験名に紐づく安定した提出履歴の作成を求められたときに使う。"
---

# Kaggle 提出監視

同梱スクリプトで Kaggle の提出状況を polling し、安定したログ行として追記する。

## 手順

1. コンペの slug を特定する。
   - ユーザーが明示した slug を優先する。
   - なければ `KAGGLE_COMPETITION`、`.kaggle_competition`、または Kaggle CLI の既定 config を使う。
2. 提出後に monitor を開始する。Kaggle API にアクセスするため、Codex tool では最初から `sandbox_permissions: "require_escalated"` と短い justification を付ける。

```bash
python .agents/skills/kaggle-submit-monitor/scripts/monitor_submission.py EXP_NAME --competition COMPETITION
```

3. scoring が長い job では、プロジェクトルートから `nohup` で実行する。

```bash
mkdir -p logs
nohup python .agents/skills/kaggle-submit-monitor/scripts/monitor_submission.py EXP_NAME --competition COMPETITION >> logs/submission_EXP_NAME.log 2>&1 &
```

4. ログパスと `tail -f` コマンドを報告する。
5. スコアが確定したら、CV/LBなど機械処理する数値を対応する実験の`metrics.json`、提出履歴を`submissions/SUBMISSIONS.md`、監視と提出の証拠を`SESSION_NOTES.md`、解釈を`result.md`へ記録し、`experiment_summary.md`を更新する。同じCV/LBを複数のMarkdownへ手作業で転記しない。

## 出力契約

スクリプトは `logs/submission_<name>.log` に書き込み、同じ内容を stdout にも出す。完了してスコア付きの提出が検出された場合、次の形式になる。

```text
[EXP_NAME] run-time: X min, status: complete, publicScore: Y, privateScore: Z
```

ユーザーが明示的に依頼しない限り、代理で submit しない。このスキルは監視と記録だけを行う。
