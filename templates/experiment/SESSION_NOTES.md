# {{ EXPERIMENT_NAME }} セッションノート

## 目的

TODO

## 現在の状態

- Route: ml_model
- 状態: 計画中
- CV: まだなし
- LB: まだなし

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 予定

```bash
task prepare-kaggle-notebooks EXP={{ EXPERIMENT_NAME }} EXTRA_ARGS="--strict"
task push-kaggle-train EXP={{ EXPERIMENT_NAME }}
task kaggle-status KERNEL=<username>/<train-kernel-slug>
task push-kaggle-infer EXP={{ EXPERIMENT_NAME }}
task kaggle-status KERNEL=<username>/<inference-kernel-slug>
```

## 変更点

- TODO

## 再現性メモ

- seed policy: TODO
- stochastic components: TODO
- CPU/GPU runtime: TODO
- Kaggle kernel id / version: TODO
- input / feature schema SHA: TODO
- feature content SHA: TODO
- model manifest / model SHA: TODO
- prediction SHA: TODO
- submission SHA: TODO
- rerun check: TODO

## 次のアクション

1. TODO
