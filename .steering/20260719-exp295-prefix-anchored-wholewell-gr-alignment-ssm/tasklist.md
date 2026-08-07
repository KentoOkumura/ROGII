# タスクリスト

## 目的

設計確定と将来の実装・実行を分離する。下記の将来タスクは実装、GPU利用、Kaggle pushの承認を意味しない。

## Stage A実装（2026-07-19承認・完了）

- [x] Jupytext percent形式のcompact self-contained train候補を別名で実装する。
- [x] mask-first horizontal loader、Type Well loader、fold/pseudo-cut freeze、neighbor-source count guardを実装する。
- [x] prefix-conditioned multi-scale encoder、structured loss、exp209 exact decoderを実装する。
- [x] real/shuffled/geometry-onlyを同一trained modelでdecodeするcontract testsを追加する。
- [x] fold 0のactive architecture 1 / seed 1 / neural model 1、LightGBM 0 / booster 0 / control再学習0を再確認する。
- [x] fail-closed compact inference候補を実装し、Stage B promotion前のinference/submissionを拒否する。
- [x] Jupytext `--test`、py_compile、Ruff、専用pytest、repository tests、`make validate-exp`を通す。
- [x] canonical notebook採用とKaggle T4 Stage A pushの別承認を得る。

## Stage A runtime contract修復（2026-07-20承認）

- [x] version 2のhard truth path infeasibleをtruth-only auditで再現し、decoder転記bugではないことを確認する。
- [x] fixed exp209 decoderを維持し、Gaussian soft-label structured likelihood `sigma=0.35 ft`へ変更する承認を得る。
- [x] hard truth one-hot pathを、通常posteriorとlabel-conditioned posteriorの差で学習するobjectiveへ置換する。
- [x] canonical train Notebook、package、記録を同期して静的・契約検証を通す。
- [x] 同じcanonical kernelへT4/run-on-pushのversion 3をpushし、開始状態を確認する。
- [x] version 3 timeoutを診断し、runtime gate FAILとしてStage B/inference/submissionをbranch closeする。

## 将来タスク（Stage A PASSかつStage B承認後）

- fold 0 model/config/SHAを再利用し、fold 1-4の4 modelsだけを追加学習する。
- pooled/fold/subgroup/by-well/negative-control/posterior metricsを集約する。
- fixed gateだけで`promotion_pass`、`architecture_signal_only`、`close`を判定する。
- result、metrics、SESSION_NOTES、experiment_summary、backlogを更新する。

## 将来タスク（Stage B promotion PASSかつStage C承認後）

- 同じexp内でcompact self-contained inferenceを実装する。
- current-test input parity、5 fold model manifest、prediction SHA、runtime、submit-checkを確認する。
- submissionを作成する場合は`kaggle-submit-check`、提出後は`kaggle-submit-monitor`を使う。

## ブロック中

- Stage B、Stage C、submissionは先行stage未通過かつ未承認。

## 次のアクション

承認済みruntime contract修復を検証し、同じcanonical kernelのversion 3でStage A fold 0を再実行する。

## 完了

- `kaggle-strategy`で既存backlog、現行anchor、関連するpositive/negative実験を確認した。
- `docs/06_reproducibility.md`を確認し、CUDA学習、fold map、pseudo-cut、SHA方針を設計へ反映した。
- exp295のbacklog優先度、neighbor-free境界、数理モデル、architecture、stage分岐、success/failure policyを固定した。
- `.steering/20260719-exp295-prefix-anchored-wholewell-gr-alignment-ssm/`を作成した。
- `experiments/exp295_prefix_anchored_wholewell_gr_alignment_ssm/` scaffoldを作成した。
- raw horizontal/Type Wellの実ファイル名とcolumn schemaを確認し、configのglob/input allowlistへ反映した。
- strict experiment validation、project template validation、YAML/JSON parse、未記入placeholder監査を通過した。
- `make update-summary`で`experiment_summary.md`へexp295を記録した。
- 設計確定時点ではscaffold Notebookを変更せず、実装・package・run・inference・submissionを行わなかった。その後、別名compact Stage A候補とfail-closed inference候補だけを追加した。
- Stage A実装後、専用pytest `9 passed, 1 skipped`、repository pytest `307 passed, 1 skipped`、Ruff、py_compile、Jupytext train/inference `--test`、strict experiment/template validationを通過した。skipはローカル環境にPyTorchがないためのexact Torch posterior/gradient testだけである。
