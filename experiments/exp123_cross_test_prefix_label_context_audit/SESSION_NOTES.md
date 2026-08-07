# exp123_cross_test_prefix_label_context_audit セッションノート

## 目的

`KAGGLE_DIRECTION.md` の `cross_test_prefix_label_context_audit` を実装する。同じ pseudo test batch 内の他 validation wells の finite `TVT_input` prefix label から、batch-level bias / slope / residual scale を診断し、target well tail の prefix-only baseline へ安定して効くかを読む。

## 現在の状態

- Route: `ml_model`
- 状態: Kaggle train v1 完了
- CV: 15.909852870734554
- LB: まだなし
- 提出: なし。inference notebook は guard。
- Kaggle kernel: `kentookumura/exp123-cross-test-prefix-label-audit-train` v1
- Kaggle output: `experiments/exp123_cross_test_prefix_label_context_audit/kaggle/output/train_v1/`

## コマンドログ

- 2026-06-25: `uv run python scripts/new_steering.py --experiment exp123_cross_test_prefix_label_context_audit` で steering docs を作成。
- 2026-06-25: `uv run python scripts/new_experiment.py --name exp123_cross_test_prefix_label_context_audit` で実験を作成。
- 2026-06-25: `config.yaml` を rules-risk diagnostic 用に更新。
- 2026-06-25: `cross_test_prefix_label_context_audit.py` を追加。
- 2026-06-25: train notebook を入力確認、監査実行、生成物 preview 構成に更新。
- 2026-06-25: inference notebook を submission を作らない guard に更新。
- 2026-06-25: `uv run python -m py_compile experiments/exp123_cross_test_prefix_label_context_audit/cross_test_prefix_label_context_audit.py experiments/exp123_cross_test_prefix_label_context_audit/settings.py` が通過。
- 2026-06-25: notebook JSON parse が通過。
- 2026-06-25: `uv run python scripts/validate_experiment.py --experiment exp123_cross_test_prefix_label_context_audit` が通過。
- 2026-06-25: `/tmp/exp123_cross_test_prefix_label_context_audit_smoke2` に `--max-wells 20` の script smoke を実行し、生成物作成まで通過。これは正式 CV ではないため `metrics.json` には記録しない。
- 2026-06-25: `uv run ruff format experiments/exp123_cross_test_prefix_label_context_audit/cross_test_prefix_label_context_audit.py` が通過。
- 2026-06-25: `uv run ruff check experiments/exp123_cross_test_prefix_label_context_audit/cross_test_prefix_label_context_audit.py` が通過。
- 2026-06-25: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp123_cross_test_prefix_label_context_audit --notebook train --run-on-push --strict --title "exp123 cross test prefix label audit train" --kernel-id kentookumura/exp123-cross-prefix-label-audit-train` が通過。
- 2026-06-25: `kaggle kernels push -p experiments/exp123_cross_test_prefix_label_context_audit/kaggle/train` で Kaggle train v1 を push。metadata id は `kentookumura/exp123-cross-prefix-label-audit-train` だったが、title slug と一致しない warning が出て、Kaggle URL / 実体は `kentookumura/exp123-cross-test-prefix-label-audit-train` として作成された。
- 2026-06-25: `kaggle kernels pull kentookumura/exp123-cross-test-prefix-label-audit-train -p /tmp/kaggle-pull/exp123-cross-test-prefix-label-audit-train-v1 -m` で存在確認。
- 2026-06-25: `kaggle kernels logs kentookumura/exp123-cross-test-prefix-label-audit-train` は初回空。`timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp123-cross-test-prefix-label-audit-train` では実行ログを取得できた。
- 2026-06-25: `kaggle kernels output kentookumura/exp123-cross-test-prefix-label-audit-train -p experiments/exp123_cross_test_prefix_label_context_audit/kaggle/output/train_v1` で output を取得。
- 2026-06-25: Kaggle output の `metrics.json` をローカル `metrics.json` に反映。

## 予定コマンド

```bash
task prepare-kaggle-notebooks EXP=exp123_cross_test_prefix_label_context_audit EXTRA_ARGS="--notebook train --run-on-push --strict"
task push-kaggle-train EXP=exp123_cross_test_prefix_label_context_audit
```

## 変更点

- target well 自身の prefix-only baseline として `hold_prefix_control`、`self_linear_prefix_control` を作る。
- target well と同じ validation fold 内の他 wells の prefix 終端付近における hold residual を集計する。
- hold 基準の cross-batch 候補として bias、slope、scale-shrunk slope、scale-shrunk bias を比較する。
- summary JSON、candidate metrics、fold metrics、by-well metrics、bucket metrics、context stats を保存する。

## 結果

- 実行: Kaggle train v1 COMPLETE
- rows: 3,783,989
- wells: 773
- best: `hold_prefix_control` RMSE 15.909852871 / MAE 11.196479702 / within10 0.578628532
- `cross_batch_bias_scale_hold`: RMSE 15.917976341
- `cross_batch_bias_hold`: RMSE 15.920967640
- `cross_batch_scale_slope_hold`: RMSE 20.375654980
- `cross_batch_slope_hold`: RMSE 24.204548712
- `self_linear_prefix_control`: RMSE 1404.728336097
- fold selection は全 fold で `hold_prefix_control` を選択し、selection RMSE 15.909852871。

## 判断

cross-test prefix label context は採用しない。他 well の visible `TVT_input` label を使う rules risk があるうえ、OOF 診断でも hold baseline を超えなかった。推論化、提出、exp092 系 feature 化には進めない。

## 再現性メモ

- seed policy: sorted file order + deterministic `GroupKFold`。
- stochastic components: なし。
- CPU/GPU runtime: CPU-only。GPU 不使用。
- Kaggle kernel id / version: `kentookumura/exp123-cross-test-prefix-label-audit-train` v1。
- input / feature schema SHA: 未記録。
- feature content SHA: no feature cache。
- model manifest / model SHA: no model。
- prediction SHA: row-level prediction は保存しない。
- submission SHA: no submission。
- rerun check: なし。診断実験として単回 Kaggle train v1 を正とする。

## 次のアクション

1. `cross_test_prefix_label_context_audit` を backlog から外す。
2. 同系統は target-free context audit / high-drift confidence feature 側へ戻す。
