# 要件

## 依頼

`projection_confidence_error_map` バックログを実装する。`exp094_projection_only_on_exp073` で global projection は OOF 全体を改善したが near-prefix を壊したため、raw/base と projected の差分を well 条件別に error map 化し、projection が効く条件と使ってはいけない条件を読む。

## 制約

- Route: `ml_model`
- 親実験: `exp094_projection_only_on_exp073`
- 参照元: `pilkwang/rogii-target-free-tvt-geosteering` の `TVT + Z - anchor` projection
- 診断専用とし、inference port や submission は作らない。
- gate 条件に使う特徴は target-free な raw geometry、GR 欠損、prefix 長、既存 PF/Beam disagreement、native typewell group に限定する。
- `target_tvt` は raw/projected の評価と error map の集計にだけ使う。
- Kaggle Notebook 実行を正とする。ローカル notebook 実行は明示的な smoke debug に限定する。
- 再現性: exp124 自体は RNG なし。gzip 入力は decompressed content SHA を記録する。

## 受け入れ基準

- `experiments/exp124_projection_confidence_error_map/` に config、train/inference notebook、補助スクリプト、記録ファイルがある。
- train notebook で exp094 best predictions を読み、raw train から Z span / GR missing / prefix length を復元できる。
- optional context として PF/Beam disagreement と native typewell group を見つかった場合に結合できる。
- 生成物として row error map、well error map、bucket metrics、gate metrics、summary JSON、生成物 README を保存する。
- `make validate-exp EXP=exp124_projection_confidence_error_map` が通る。
- `make prepare-kaggle-notebooks EXP=exp124_projection_confidence_error_map EXTRA_ARGS="--notebook train --run-on-push --strict"` が通る。
- inference notebook は no-submission summary のみを書き、提出候補を作らない。
