# exp093_pf_candidate_coverage_then_ranker_audit 結果

## 状態

Kaggle train v1 完了。

## 仮説

PF/Beam/likelihood-PF 候補集合が真値近傍候補を十分に含んでいるなら、次段で直接 TVT regression ではなく候補 ranker / N-way classifier として学習できる。coverage が不足する bucket では、ranker 学習ではなく候補生成側の失敗条件として閉じる。

## 設定

- 親: `exp091_self_gr_likelihood_pf_beam_probe`
- cache 親: `exp072_exp063_full_replay_feature_cache`
- 診断参照: `exp056_public_sel15_pf_oof_multicutoff_artifact`、`exp083_pf_beam_true_tvt_2d_well_eda`、`exp087_prefix_backtest_tvt_confidence`
- 検証: train well pseudo-tail candidate coverage audit
- メトリック: candidate RMSE、within 1/2/5/10 ft、oracle topK coverage、target-free rank-score topK coverage、bucket weak coverage
- シード: exp093 内では新規乱数なし

## 結果

- Kernel: `kentookumura/exp093-pf-candidate-coverage-ranker-train` v1
- rows: 3,783,989
- wells: 773
- runtime: 3,851.92 sec
- source cache SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- source decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- schema SHA: `700d38149f583c3ab6574ea7b163c3c8709c2514b675bea381d822f82f4809b8`

best single candidate は `likpf_mean` で、RMSE 11.594897 / MAE 7.067633 / within10 0.772807。

candidate set oracle は以下。

| candidate set | RMSE | MAE | within10 | selected self-GR rate |
| --- | ---: | ---: | ---: | ---: |
| baseline_primary | 7.434030 | 3.745228 | 0.906525 | 0.000000 |
| baseline_plus_self_gr | 6.958935 | 3.335769 | 0.922492 | 0.155758 |

self-GR 追加により oracle RMSE は -0.475095、within10 は +0.015967 改善した。ただし設定上の `min_oracle_rmse_gain=1.0` には届かない。

target-free rank score top1 は以下。

| candidate set | RMSE | MAE | within10 | selected self-GR rate |
| --- | ---: | ---: | ---: | ---: |
| baseline_primary | 12.507841 | 7.626705 | 0.748803 | 0.000000 |
| baseline_plus_self_gr | 29.985529 | 8.918838 | 0.746819 | 0.004808 |

候補集合には headroom があるが、現行 rank score は oracle headroom を活かせていない。

## 候補選択回数

`candidate_long.csv.gz` から、各候補が oracle best になった回数と target-free rank score top1 になった回数を集計した。保存先は `kaggle/output/train_v1/artifacts/exp093_pf_candidate_coverage_then_ranker_audit_selection_counts.csv`。

| candidate | oracle best count | oracle best rate | rank score top1 count | rank score top1 rate | 解釈 |
| --- | ---: | ---: | ---: | ---: | --- |
| `likpf_mean` | 1,242,769 | 0.328428 | 2,729,282 | 0.721271 | 現行 rank score の主選択候補で、oracle でも最多 |
| `pf_ancc` | 1,092,069 | 0.288603 | 0 | 0.000000 | oracle では非常に重要だが、現行 rank score では一度も top1 にならない |
| `beam_mean` | 443,268 | 0.117143 | 1,036,513 | 0.273921 | rank score でも選ばれる主要候補 |
| `last_anchor_tvt` | 370,631 | 0.097947 | 0 | 0.000000 | 近距離などでは oracle best になるが、rank 対象としては選ばれない |
| `self_gr_sc25` | 175,030 | 0.046255 | 0 | 0.000000 | self-GR 系で oracle contribution が最も大きいが rank score では未選択 |
| `self_gr_best` | 167,674 | 0.044311 | 0 | 0.000000 | 局所的な oracle headroom あり |
| `self_gr_sc15` | 141,924 | 0.037506 | 0 | 0.000000 | 局所的な oracle headroom あり |
| `hyb` | 73,739 | 0.019487 | 0 | 0.000000 | 単体は弱いが oracle best になる場面あり |
| `sc_ens` | 49,871 | 0.013179 | 0 | 0.000000 | 単体は弱いが oracle best になる場面あり |
| `self_gr_sc8` | 24,476 | 0.006468 | 0 | 0.000000 | 小さいが oracle best になる場面あり |
| `self_gr_ens` | 2,538 | 0.000671 | 18,194 | 0.004808 | oracle contribution は小さいが rank score で少数選ばれる |

「rank score で一度も top1 に選ばれず、oracle best にも一度もならない」という意味での完全なノイズ候補はなかった。

一方で、`pf_ancc` は oracle best が 1,092,069 rows あるのに rank score top1 が 0 rows であり、現行 rank score が PF ANCC を過小評価している。`self_gr_sc15` / `self_gr_sc25` / `self_gr_best` も oracle contribution はあるが、現行 score では top1 に選べていない。

## 再現性

- deterministic anchor: false
- seed policy: no new RNG in exp093
- kernel version: `kentookumura/exp093-pf-candidate-coverage-ranker-train` v1
- feature content SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- model SHA / manifest SHA: model なし
- prediction SHA: prediction なし
- submission SHA: submission なし
- rerun result: train-side audit のため未実施

## 解釈

summary JSON の `ranker_readiness.recommendation` は `ranking_or_likelihood_scorer_audit_before_ranker`。

候補集合そのものは有望で、baseline + self-GR の oracle within10 は 0.922492 まで届く。しかし現行 target-free rank score は `likpf_mean` と `beam_mean` に偏り、`pf_ancc` や self-GR scale candidates の oracle contribution を拾えていない。

したがって、次は候補生成を増やすよりも、PF ANCC を正しく上位化する scorer / likelihood calibration / supervised candidate ranker の小実験を優先する。

## 次

`pf_candidate_ranker_or_nway_classifier` に進む前に、現行 rank score の失敗、特に `pf_ancc` を一度も top1 にしない問題を修正する scorer audit を行う。
