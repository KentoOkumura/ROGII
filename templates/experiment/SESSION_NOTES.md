# {{ EXPERIMENT_NAME }} セッションノート

## 目的

TODO

## 現在の作業

- 作業内容: TODO
- ブロック要因: なし
- 次: TODO

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 予定

共通確認を先に行います。

```bash
task validate-exp EXP={{ EXPERIMENT_NAME }}
task check-exp EXP={{ EXPERIMENT_NAME }}
task test-exp EXP={{ EXPERIMENT_NAME }}
```

次のうち、実験契約に必要なnotebookだけを予定へ残します。学習を伴わないauditやdiagnosticではtrain、提出を目的としない実験ではinferenceを機械的に実行しません。

trainが必要な場合:

```bash
task prepare-kaggle-notebooks EXP={{ EXPERIMENT_NAME }} EXTRA_ARGS="--notebook train --run-on-push"
task push-kaggle-train EXP={{ EXPERIMENT_NAME }}
task kaggle-logs KERNEL=<generated-kernel-id>
```

inferenceが必要な場合:

```bash
task prepare-kaggle-notebooks EXP={{ EXPERIMENT_NAME }} EXTRA_ARGS="--notebook inference --run-on-push"
task push-kaggle-infer EXP={{ EXPERIMENT_NAME }}
task kaggle-logs KERNEL=<generated-kernel-id>
```

prepare後に生成された`kernel-metadata.json`からkernel idを取得して`KERNEL`へ指定する。id末尾とtitle由来slugが一致し、50文字以内であることを確認する。自動生成では上限や衝突を解消できない場合だけ、`kaggle-platform`の規則に従って意味のある短縮id/titleを明示する。placeholderのまま実行しない。

## 変更点

- TODO

設定と再現性方針は`config.yaml`、kernel情報、Kaggle Notebook実行時間、生成物SHA、rerun比較、実験statusは`metrics.json`へ記録する。このファイルには、それらを得たコマンド、時刻、途中経過、失敗と修正を時系列で残す。提出した場合はsubmission ref・提出日時・submission scoring status・監視開始からscore確定までの所要時間を、詳細な時系列の正として記録し、Notebook実行時間と混同しない。`SUBMISSIONS.md`には横断比較に必要な最終スナップショットだけを`record-submission`で記録し、状態遷移、観測時刻、実行コマンドを転記しない。

## 次のアクション

1. TODO
