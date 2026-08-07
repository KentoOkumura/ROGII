# exp123_cross_test_prefix_label_context_audit

## 状態

- ルート: `ml_model`
- 状態: completed
- CV: 15.909852870734554
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-25
- 親実験: `exp037_test_time_prefix_online_training_audit`

## 仮説

同じ pseudo test batch 内の他 well に見えている finite `TVT_input` prefix label から、batch-level の residual bias / slope / scale を読めるなら、target well tail の prefix-only baseline に対する系統ずれを診断できる可能性がある。

ただし他 well の `TVT_input` を label として使うため rules risk があり、改善しても提出候補にはしない。

## 変更点

- train wells を `GroupKFold` で pseudo test batch に分ける。
- target well ごとに、同じ validation fold 内の他 wells の prefix label だけを context source にする。
- `hold_prefix_control` と `self_linear_prefix_control` を基準に、hold 基準の cross-batch bias / slope / scale shrink 候補を比較する。
- inference notebook は guard とし、`submission.csv` を生成しない。

## 検証方針

- Fold: `GroupKFold(n_splits=5)`
- Group: well id
- Stratification: なし
- Leakage Check: target well evaluation rows の true `TVT` は scoring のみに使う。target well の context 推定では target well 自身を除外する。他 validation wells の visible `TVT_input` prefix label 利用は意図的な rules-risk 診断として記録する。

## 実行入口

- 学習 notebook: `exp123_cross_test_prefix_label_context_audit_train.ipynb`
- 推論 notebook: `exp123_cross_test_prefix_label_context_audit_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp123_cross_test_prefix_label_context_audit EXTRA_ARGS="--notebook train --run-on-push --strict"`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は明示的な smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- Kaggle train v1 が 773 wells / 3,783,989 rows で完了した。
- context 推定から target well 自身を除外する診断フローと生成物保存は動作した。

### 悪かった点

- 最良は `hold_prefix_control` のままで、cross-batch prefix label 補正は全体 RMSE を改善しなかった。
- `cross_batch_bias_scale_hold` は 15.917976、`cross_batch_bias_hold` は 15.920968 で hold baseline 15.909853 より悪化した。
- slope 系はさらに悪く、`cross_batch_scale_slope_hold` 20.375655、`cross_batch_slope_hold` 24.204549 だった。

### リスク / 注意

- 他 well の visible `TVT_input` label を使うため、organizer rules / leakage 解釈の確認なしに推論化しない。
- exp036/exp037 では prefix 由来の same-OOF 改善が holdout に転移しなかったため、global RMSE だけで判断しない。

## 次

1. cross-test prefix label context は推論化しない。
2. 他 well の visible `TVT_input` label 利用は、rules risk に加えて OOF 上も支持されなかったため閉じる。
3. same-batch 文脈は target-free covariate context / high-drift confidence feature 側を優先する。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
