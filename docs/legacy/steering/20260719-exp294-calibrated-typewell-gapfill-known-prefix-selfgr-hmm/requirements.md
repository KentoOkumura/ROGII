# 要件

## 依頼

`exp223_joint_typewell_self_gr_hmm_likelihood_probe` の self-GR 参照曲線について、known `TVT_input` 区間の欠損 GR だけを Type Well から復元した GR で補完し、HMM の精度と安全性が改善するかを検証できる実験を設計する。

この steering 作成時点では、バックログ、実験ディレクトリ、設定、検証契約だけを確定する。補完器、HMM、notebook の実装、Kaggle 実行、inference、submission は行わない。

## 仮説

`exp223` は self-GR donor window 内の生 GR 欠損を線形補間している。known `TVT_input` から Type Well GR をサンプルし、well ごとに observed known GR へ頑健 affine 校正した値で欠損セルだけを埋めれば、観測 GR を変更せずに donor motif の局所形状を改善できる。その結果、固定した self-GR likelihood が欠損由来の偽ピークへ引かれにくくなる可能性がある。

## 制約

- Route: `ensemble`。`exp209` の Type Well exact HMM と `exp223` の same-well self-GR likelihood の両方が予測生成に本質的に寄与するため、親実験と同じ route を使う。
- 親実験: `exp223_joint_typewell_self_gr_hmm_likelihood_probe`。
- 反証参照: `exp225_state_known_tvt_self_gr_hmm_emission`。state ごとの known-curve boost は RMSE 14.212954500、1000+ `+2.931795 ft`、worst well `+49.423573 ft` だったため、Type Well 復元 GR を target state の直接 emission や replacement には使わない。
- observed known-prefix GR は bitwise に変更しない。変更対象は raw GR が非有限で、同じ行の `TVT_input` が有限かつ Type Well TVT 範囲内の donor セルだけとする。
- self-GR anchor center、window eligibility、missing-rate gate は元の raw missing mask を使い、Type Well 補完によって新しい anchor を増やさない。
- target / unknown suffix の self-GR receiver、base Type Well HMM emission、HMM transition、grid、band、calibration は `exp223` から変更しない。target 側 Type Well gap-fill 件数は常に 0 とする。
- Type Well 範囲外は外挿しない。補完不成立セルは `exp223` の既存線形補間へ戻す。
- active self-GR variant は `alpha=0.07`、`clip=1.0`、`boost_only` の1本だけとする。alpha、clip、window、affine、threshold の grid は行わない。
- 保存済み `exp223` control を使い、親/controlを再実行しない。
- Stage 0 / Stage 1 とも CPU-only、LightGBM config 0、trained fold 0、booster 0。Stage 1 は1 HMM variant、773 well-runs。
- 再現性は `docs/06_reproducibility.md` に従う。擬似欠損位置は Python の `hash()` や global RNG ではなく stable SHA256 で固定し、gzip は decompressed content SHA を主証拠にする。
- true suffix `TVT` は予測、補完、affine fit、mask作成、gate選択に使わず、全 prediction / feature SHA freeze 後の評価 join にだけ使う。

## 受け入れ基準

### 設計完了

- `docs/legacy/steering/20260719-exp294-calibrated-typewell-gapfill-known-prefix-selfgr-hmm/` に requirements / design / tasklist があり、変更境界、二段階ゲート、リーク防止、再現性、禁止事項が確定している。
- `experiments/exp294_calibrated_typewell_gapfill_known_prefix_selfgr_hmm/` に design-only の config / README / SESSION_NOTES / result / metrics があり、実装・実行済みと誤認できる記述がない。
- `KAGGLE_DIRECTION.md` の未着手バックログに、現行の高優先候補より下の「低-中・Stage 0 条件付き」として記録されている。
- `experiment_summary.md` に design-only / 未実装として記録されている。

### Stage 0: 擬似欠損信号監査

- control は残存 observed GR の線形補間、variant は同じ入力から作る校正済み Type Well gap-fill とする。
- held-out GR pooled RMSE が control 比 5%以上改善する。
- 有効 block の ZNCC が pooled で `+0.02` 以上改善する。
- RMSE 改善と ZNCC 正方向をそれぞれ5 reporting folds中4 folds以上で満たす。
- by-well RMSE の p95 が control より悪化しない。
- observed known GR exact parity、raw missing mask parity、target-side Type Well fill 0、fit時の held-out row 除外をすべて満たす。
- 1項目でも失敗した場合は Stage 1 を実装・実行せず、この仮説を閉じる。

### Stage 1: 固定 self-GR HMM 比較

- Stage 0 全条件 PASS 後、かつ別途ユーザー承認後だけ実装・実行する。
- primary baseline `exp223` RMSE 11.349950650 から `0.10 ft` 以上改善し、RMSE `<=11.249950650` とする。
- stable well-hash reporting folds の4/5以上で `exp223` より `0.10 ft` 以上改善する。
- `1000_plus`、exp115 verification-like spatial、typewell-purged は `exp223` 比 `+0.02 ft` 以内とする。
- worst-well delta RMSE は `+0.25 ft` 以下とする。
- natural known-prefix GR missing rate 0 の well は prediction 最大絶対差 `<=1e-6 ft`、missing rate 1%以下の scope は RMSE delta `<=+0.02 ft` とする。
- 上記を満たしても、`exp209` HMM/likPF blend 10.269696 に達しない限り raw-test port / inference / submission へ自動昇格しない。昇格には別途設計とユーザー承認を必要とする。

deterministic anchor として扱う場合は、input / feature schema / feature content / prediction / submission / model manifest の該当 SHA と Kaggle kernel version を記録する。ただし本実験の初回 train-side audit は deterministic submission anchor ではない。
