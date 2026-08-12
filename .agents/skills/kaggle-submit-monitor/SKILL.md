---
name: kaggle-submit-monitor
description: "`kaggle competitions submit` 後の特定の Kaggle submission ref を監視する。特に scoring が長い code competition 向け。監視開始からscore確定までの経過時間、LB/public score、submission log の tail、実験名とsubmission refを対応付けた提出履歴の記録を求められたときに使う。"
---

# Kaggle 提出監視

同梱スクリプトで Kaggle の提出状況を polling する。pollingログは一時生成物であり、Gitへ保存しない。最終的な提出事実は実験記録と`SUBMISSIONS.md`へ残す。

## 手順

1. コンペの slug を特定する。
   - ユーザーが明示した slug を優先する。
   - 明示指定がなければ、リポジトリルートの`project.yml`にある`competition.slug`を使う。未設定なら停止し、Kaggle CLIの既定configから暗黙に補完しない。
2. submit前の提出一覧と、submit後の提出一覧またはsubmit結果を比較し、今回作成されたsubmission refを一意に特定する。message、提出日時、kernel id/versionを照合し、候補が複数ある場合は停止する。単に一覧の先頭を対象にしない。
3. 特定したrefを`--submission-ref`で固定してmonitorを開始する。Kaggle API にアクセスするため、Codex tool では最初から `sandbox_permissions: "require_escalated"` と短い justification を付ける。

```bash
uv run python .agents/skills/kaggle-submit-monitor/scripts/monitor_submission.py EXP_NAME \
  --submission-ref SUBMISSION_REF
```

`project.yml`とは別のコンペを監視する場合だけ`--competition COMPETITION`で上書きする。

4. scoring が長い job では、プロジェクトルートから `nohup` で実行する。実験名が`experiments/`配下のディレクトリ名と一致する場合、既定の監視ログは無視対象の`experiments/<exp>/artifacts/submission-monitor.log`へ保存される。別名で監視するときは`--log-file`で対象実験の`artifacts/`を明示する。

```bash
nohup uv run python .agents/skills/kaggle-submit-monitor/scripts/monitor_submission.py EXP_NAME \
  --submission-ref SUBMISSION_REF \
  --log-file experiments/EXP_NAME/artifacts/submission-monitor.log \
  >/tmp/submission_EXP_NAME.runner.log 2>&1 &
```

5. 一時ログのパスと`tail -f`コマンドを報告する。一時ログをGitへ追加しない。
6. スコアが確定したら、まず`task record-exp EXP=expXXX PUBLIC_LB=...`（Private LB判明後は`PRIVATE_LB=...`も指定）を実行する。次に`task record-submission EXP=expXXX SUBMISSION=/path/to/submission.csv SUBMISSION_REF=...`を実行し、同じrefは既存行を更新する。code competitionのoutputをローカル取得していない場合は、Kaggle側で対象ファイルを確認してから`SUBMISSION=submission.csv EXTRA_ARGS="--allow-missing-file"`を指定する。監視固有のsubmission ref、scoring status、score確定までの所要時間と解釈は`AGENTS.md`の役割分担に従って一度だけ記録し、Kaggle Notebook実行時間とsubmission scoring所要時間を混同しない。

## 出力契約

スクリプトは`--submission-ref`と一致する行だけを監視する。refが見つからなくても最新行へ切り替えず、見つかるまで待つ。実験の`artifacts/`または明示した`--log-file`へ一時ログを書き込み、同じ内容をstdoutにも出す。実験を特定できない場合は`/tmp/kaggle-submission-monitor/`へ保存する。完了してスコア付きの提出が検出された場合、次の形式になる。

```text
[EXP_NAME] scoring-elapsed: X min, submission-status: complete, publicScore: Y, privateScore: Z
```

ユーザーが明示的に依頼しない限り、代理で submit しない。このスキルは監視と記録だけを行う。
