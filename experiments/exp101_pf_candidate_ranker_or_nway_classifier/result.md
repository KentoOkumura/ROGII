# exp101_pf_candidate_ranker_or_nway_classifier 結果

## 状態

Kaggle train v1 完了。結論は不採用。

## 仮説

PF/Beam/likelihood-PF 候補集合には oracle headroom があるが、target-free scorer は重要な `pf_ancc` を拾えていない。exp099 の multi-observation likelihood features を supervised ranker の特徴量として使えば、`likpf_mean` 単体および target-free scorer より良い候補 index を選べる可能性がある。

## 設定

- 親: `exp099_pf_multi_observation_likelihood_probe`
- 入力: exp099 v2 wide train feature cache
- 候補: `pf_ancc`, `beam_mean`, `likpf_mean`, `sc_ens`, `hyb`
- 検証: GroupKFold by `well`
- 学習器:
  - LightGBM multiclass
  - candidate-long binary scorer
  - candidate-long predicted-error ranker

## 判定基準

提出候補化はしない。次へ進む条件は、OOF で `likpf_mean` 単体 RMSE 11.594897 と exp093 baseline rank score RMSE 12.507841 を明確に超え、かつ `pf_ancc` を一定数選べること。改善時も path switch、worst-well、distance bucket、raw-test feature parity を別実験で確認する。

## 実装検証

- `py_compile`: 成功
- `ruff check experiments/exp101_pf_candidate_ranker_or_nway_classifier`: 成功
- `make validate-exp EXP=exp101_pf_candidate_ranker_or_nway_classifier`: 成功
- exp099 v2 cache dry smoke: 20,000 rows / 5 wells / 47 features / candidate-long 5,000 rows の生成に成功
- Kaggle train package 生成: 成功
- local `.venv` には `lightgbm` がないため、ローカルでの学習 smoke は未実行

## 次

static supervised ranker は `likpf_mean` 単体を超えず、row-wise path switch も多いため、この方向は提出候補化しない。

## Kaggle train v1

- Kernel: `kentookumura/exp101-pf-cand-ranker-train` v1
- URL: `https://www.kaggle.com/code/kentookumura/exp101-pf-cand-ranker-train`
- status: `COMPLETE`
- runtime: 2,618.39 sec
- rows: 3,783,989
- wells: 773
- output: `kaggle/output/train_v1`
- GPU: false
- internet: false

## 結果

| variant | mode | RMSE | MAE | within10 | oracle label accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `oracle` | oracle | 7.434030 | 3.745228 | 0.906525 | 1.000000 |
| `likpf_mean_single` | baseline | 11.594898 | 7.067633 | 0.772807 | 0.385230 |
| `lgb_candidate_error_ranker` | oof | 11.600097 | 7.006913 | 0.771452 | 0.389880 |
| `lgb_candidate_binary` | oof | 11.814474 | 7.194226 | 0.759631 | 0.431650 |
| `lgb_multiclass` | oof | 12.106697 | 7.336443 | 0.756271 | 0.430136 |
| `multiobs_score_top1` | baseline | 89.994391 | 38.086730 | 0.523815 | 0.235524 |

best OOF は `lgb_candidate_error_ranker` だが、`likpf_mean_single` より RMSE が `+0.005199` 悪い。summary の recommendation は `ranker_not_supported`。

## 選択分布と path continuity

best OOF の `lgb_candidate_error_ranker` は `pf_ancc` を 35.98% 選ぶため、`pf_ancc` を全く拾えない問題自体は改善した。しかし `likpf_mean` 単体より RMSE / within10 が改善せず、worst well では RMSE 58.75 まで悪化する。

best OOF の最大 `path_switch_per_1000_rows` は 365.996。row-wise selector として切替が多すぎるため、そのまま推論化する根拠はない。

## 再現性

- exp099 input raw SHA: `4bd9df60f5c09f7a3029dac399afef73aa45b0158a7fd06a62a56f85fd0fde38`
- exp099 input decompressed SHA: `1939d536b1e56f7c0ea3847cc386ef769b0d33759d16e816c9ce180f0532df9a`
- schema SHA: `203e4f9a280fe901f5f21d39b85c3e0e2a7fe10c466081c15015c7fb014a0413`
- output prediction decompressed SHA: `05cd6bc1658ab4e7c2958154bf9358582a6f1f38932ec7db8637613c00d6d09a`
- model manifest SHA: `4f453761f1cc09042767baa934f8a1c5a89036bfb1c244a5f3fc5ab0cc843cc5`

## 解釈

候補集合には oracle headroom があるが、row-wise supervised selector は `likpf_mean` 単体を超えられなかった。`pf_ancc` を一定数選べても、選び方が十分に精密ではなく、worst-well と path switch を悪化させる。

したがって `pf_candidate_ranker_or_nway_classifier` は完了・不採用。次に候補選択を続けるなら、row-wise selector ではなく、segment-level / continuity-constrained selector を診断として扱う。ただし優先度は `typewell_neighbor_prior_features` や projection / native-overlap 系のほうが高い。
