# exp062_ravaghi_vs_pixiux_lgbm_features セッションノート

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
task prepare-kaggle-notebooks EXP=exp062_ravaghi_vs_pixiux_lgbm_features EXTRA_ARGS="--strict"
task push-kaggle-train EXP=exp062_ravaghi_vs_pixiux_lgbm_features
task kaggle-status KERNEL=<username>/<train-kernel-slug>
task push-kaggle-infer EXP=exp062_ravaghi_vs_pixiux_lgbm_features
task kaggle-status KERNEL=<username>/<inference-kernel-slug>
```

## 変更点

- TODO

## 次のアクション

1. TODO
