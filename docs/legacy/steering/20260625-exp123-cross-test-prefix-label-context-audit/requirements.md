# 要件

## 依頼

`KAGGLE_DIRECTION.md` のバックログ `cross_test_prefix_label_context_audit` を `exp123_cross_test_prefix_label_context_audit` として実装する。

同じ pseudo test batch 内の他 validation wells に見えている finite `TVT_input` prefix label を使い、target well の評価 tail に対して batch-level bias / slope / residual scale の診断補正が効くかを読む。

## 制約

- Route: `ml_model`
- 診断専用。提出、inference port、hidden test 用処理の採用は行わない。
- target well 自身の evaluation rows の true `TVT` は scoring のみに使う。
- target well の context 推定では、その target well 自身の prefix label も除外し、他 validation wells の visible prefix label のみ使う。
- organizer rules / leakage 解釈のリスクがあるため、改善しても submit 候補にしない。
- 先行する target-free context audit より優先度は低い。これはルール確認用の後段診断として扱う。
- 再現性: stochastic 処理、PF/Beam、GPU 学習は使わない。Kaggle bootstrap と生成物記録の扱いを設計に明記する。

## 受け入れ基準

- `docs/legacy/steering/`、`experiments/exp123_cross_test_prefix_label_context_audit/`、`config.yaml`、train / inference notebook、`SESSION_NOTES.md`、`result.md`、`metrics.json` が揃っている。
- train notebook が、入力確認、pseudo batch audit 実行、candidate metrics / by-fold / by-well / bucket / context stats の保存まで追える構成になっている。
- inference notebook は guard とし、`submission.csv` を作らない。
- 候補は少なくとも `hold_prefix_control`、`self_linear_prefix_control`、hold 基準の cross-batch bias、cross-batch slope、cross-batch scale shrink を比較する。
- 結果生成物として summary JSON、candidate metrics、fold metrics、by-well metrics、bucket metrics、context stats を保存する。
- deterministic anchor として扱わない。model SHA / submission SHA は不要で、no-model / no-submission として記録する。
