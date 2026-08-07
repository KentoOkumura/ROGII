# exp099_pf_multi_observation_likelihood_probe 結果

## 状態

Kaggle train v1 完了。

## 仮説

exp093 の target-free rank score は `likpf_mean` / `beam_mean` に偏り、oracle contribution が大きい `pf_ancc` を上位化できなかった。既存候補 TVT を prefix TVT 位置へ写し、評価 row 周辺の複数 GR 観測点と比較する likelihood を使えば、候補の target-free 順位が改善する可能性がある。

## 設定

- 親: `exp093_pf_candidate_coverage_then_ranker_audit`
- cache 親: `exp072_exp063_full_replay_feature_cache`
- 検証: train well pseudo-tail candidate likelihood audit
- 候補: `pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb`
- 追加候補: `multiobs_top1`、`multiobs_softmax_t0p15`、`multiobs_softmax_t0p3`、`likpf_multiobs_blend_w0p25`、`likpf_multiobs_blend_w0p5`
- 観測 offset: `[-24, -12, 0, 12, 24]`

## 結果

- Kernel: `kentookumura/exp099-pf-multiobs-likelihood-train` v1
- rows: 3,783,989
- wells: 773
- runtime: 1,782.42 sec
- output: `experiments/exp099_pf_multi_observation_likelihood_probe/kaggle/output/train_v1`

best single candidate は既存 `likpf_mean` で、RMSE 11.594897 / MAE 7.067633 / within10 0.772807。

candidate set oracle は以下。

| candidate set | RMSE | MAE | within10 | selected multiobs rate |
| --- | ---: | ---: | ---: | ---: |
| baseline_primary | 7.434030 | 3.745228 | 0.906525 | 0.000000 |
| baseline_plus_multiobs | 6.897510 | 3.203476 | 0.922941 | 0.175083 |

multiobs 追加により oracle RMSE は -0.536520、within10 は +0.016415 改善した。候補集合の headroom は増えた。

一方で、target-free rank score top1 は以下。

| candidate set | RMSE | MAE | within10 | selected top |
| --- | ---: | ---: | ---: | --- |
| baseline_primary | 89.994392 | 38.086731 | 0.523815 | `beam_mean` |
| baseline_plus_multiobs | 89.994392 | 38.086731 | 0.523815 | `beam_mean` |

現行 rank score は `beam_mean` 偏重で、`likpf_mean` 単体や exp093 の baseline rank score より大きく悪化した。

multiobs 由来の単体候補も弱い。最良の `likpf_multiobs_blend_w0p25` は RMSE 25.110830、`multiobs_top1` は RMSE 89.994392。したがって、multiobs 候補を直接置換・softmax blend・target-free scorer として使う方針は不採用。

## 再現性

- deterministic anchor: false
- seed policy: no new RNG in exp099
- kernel version: `kentookumura/exp099-pf-multiobs-likelihood-train` v1
- source cache SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- source decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- schema SHA: `700d38149f583c3ab6574ea7b163c3c8709c2514b675bea381d822f82f4809b8`
- model SHA / manifest SHA: model なし
- prediction SHA: prediction なし
- submission SHA: submission なし

## 解釈

summary JSON の `probe_decision.recommendation` は `multi_observation_likelihood_supported_for_ranker_features`。ただし、これは oracle headroom の増加を根拠にしたもので、現行 target-free rank score top1 は支持しない。

結論として、multi-observation likelihood は「候補集合に当たり候補を追加する材料」としては有効だが、「そのまま候補を選ぶ scorer」としては失敗。次は `pf_candidate_ranker_or_nway_classifier` または `learned_pf_observation_likelihood_probe` に吸収し、multiobs score / MAE / NCC を supervised ranker feature として使う。

## 追記: feature cache notebook

v1 output は exp072 と同じ wide feature cache 形式ではなかったため、train notebook を更新し、次の cache 生成物を追加保存するようにした。

- `exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_train_features.csv.gz`
- `exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_feature_schema.csv`

train features は `id`, `well`, `target` をメタ列にし、source columns、既存 candidate absolute TVT、multiobs score / MAE / NCC / generated candidate を float32 feature として持つ。feature schema は exp072 と同じ `variant`, `feature_index`, `feature` 形式。

この cache 追加版は Kaggle train v2 で完了した。

- rows: 3,783,989
- wells: 773
- feature_count: 40
- train feature cache: `artifacts/exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_train_features.csv.gz`
- feature schema: `artifacts/exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_feature_schema.csv`
- train feature raw SHA256: `4bd9df60f5c09f7a3029dac399afef73aa45b0158a7fd06a62a56f85fd0fde38`
- train feature decompressed SHA256: `1939d536b1e56f7c0ea3847cc386ef769b0d33759d16e816c9ce180f0532df9a`
- schema SHA256: `203e4f9a280fe901f5f21d39b85c3e0e2a7fe10c466081c15015c7fb014a0413`

## 次

生成された wide feature cache を `pf_candidate_ranker_or_nway_classifier` の入力にする。
