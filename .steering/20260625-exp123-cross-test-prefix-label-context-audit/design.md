# 設計

## アプローチ

train horizontal wells を `GroupKFold` で validation fold に分け、各 validation fold を pseudo test batch とみなす。target well ごとに、その fold 内の他 validation wells の finite `TVT_input` prefix だけから context residual を集計する。

基準予測は、target well 自身の prefix だけで作る。

- `hold_prefix_control`: 最後の finite `TVT_input` を tail 全体に延長する。
- `self_linear_prefix_control`: target well 自身の prefix で `TVT_input ~ MD` の線形近似を作り、tail に外挿する。

cross-test prefix label context は、他 validation wells の prefix 終端付近で `TVT_input - hold_prefix_prediction` を計算し、prefix 終端からの normalized MD distance に対する residual として次の候補を作る。

- `cross_batch_bias_hold`: 他 well prefix residual の robust median bias を target hold prediction に足す。
- `cross_batch_slope_hold`: 他 well prefix residual を prefix 終端からの normalized MD distance `u` に対して線形 fit し、target tail の `u` に外挿する。
- `cross_batch_scale_slope_hold`: 他 well prefix residual scale が大きい batch では slope correction を縮める。
- `cross_batch_bias_scale_hold`: bias correction を residual scale で縮める。

## 実験範囲

- 対象実験: `exp123_cross_test_prefix_label_context_audit`
- Route: `ml_model`
- 親実験: `exp037_test_time_prefix_online_training_audit`
- 変更する変数: 他 validation wells の visible `TVT_input` prefix label から作る batch-level bias / slope / residual scale。
- 固定する変数: raw train data、GroupKFold、target well 自身の prefix-only baseline、score rows は `TVT_input` NaN tail。

## 再現性設計

- seed policy: `GroupKFold` と sorted file order による deterministic audit。乱数は使わない。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。
- 並列処理と乱数の関係: 並列処理なし。
- CPU/GPU runtime と deterministic flags: CPU-only。GPU 不使用。
- train cache / test feature regeneration の SHA 記録方針: 今回は no-model / no-feature-cache。生成物 CSV / JSON を Kaggle output として保存する。
- model manifest / prediction / submission SHA 記録方針: model なし、submission なし。row-level prediction は保存せず、metrics と context stats を保存する。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks --notebook train --run-on-push --strict` で bootstrap 付き notebook を生成してから push する。

## リスク

- リークリスク: 他 well の finite `TVT_input` を label として使うため、organizer rules / leakage 解釈のリスクがある。改善しても推論化しない。
- CV/LB 不一致リスク: pseudo test batch の他 well prefix label が hidden test と同じルールで使える保証がない。Public LB への直接転用はしない。
- ランタイム/メモリリスク: raw train CSV を読むだけでモデル学習なし。row-level prediction は保存しないため低い。
- 再現性リスク: 入力 train directory と sklearn `GroupKFold` version が固定なら deterministic。
