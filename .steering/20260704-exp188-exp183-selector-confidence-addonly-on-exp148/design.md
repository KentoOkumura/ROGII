# 設計

## アプローチ

exp148 の full-train learned-likelihood LightGBM flow を維持し、exp183 の train-side OOF best Viterbi selected path を add-only confidence feature として追加する。

exp183 は exp157/158 selector に exp181 cluster-outlier prior confidence features を加えた PF/Beam candidate selector audit で、best Viterbi RMSE 10.601481774 を記録した。一方で exp183 自体は train-side audit であり raw-test inference port は未実装なので、exp188 の初期実装では train-side OOF feature としてのみ評価する。

## 実験範囲

- 対象実験: `exp188_exp183_selector_confidence_addonly_on_exp148`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- selector 親: `exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector`
- 変更する変数: exp183 OOF selected path 由来 feature group `exp183_selector_confidence` を add-only する。
- 固定する変数: exp148 の base feature surface、exp145 learned likelihood features、GroupKFold by well、LightGBM family、GPU mode。

## Feature 設計

`exp183` OOF predictions の best Viterbi rows から、leakage 列を除外して以下を生成する。

- selected candidate code / family code / default-likPF flag / dense-family flag
- selected TVT minus last-known TVT
- selected TVT minus PF/Beam/dense candidate values
- candidate TVT spread / range
- selected path jump, local switch flag, segment length, distance to segment boundary
- exp148 OOF prediction artifact が見つかる場合のみ selected TVT minus exp148 OOF prediction

使わない列:

- `true_tvt`
- `abs_error`
- `oracle_candidate`
- `oracle_label`

## 再現性設計

- seed policy: exp148 と同じ GroupKFold seed 42、LightGBM seed family、GPU deterministic flags。
- stochastic 処理: 新規 feature merge 自体に RNG はない。GPU LightGBM は bitwise deterministic anchor とは扱わない。
- PF/Beam / likelihood-PF / seed bagging: exp188 内では再生成しない。exp072、exp145、exp183 の固定 upstream output を読む。
- 並列処理と乱数の関係: feature merge は deterministic。LightGBM は `gpu_use_dp=true`、`deterministic=true`、`force_col_wise=true`、`n_jobs/num_threads=8`。
- train cache / test feature regeneration の SHA 記録方針: train output summary に input/output SHA を記録する。raw-test regeneration は初期対象外。
- model manifest / prediction / submission SHA 記録方針: train では model manifest と OOF prediction SHA を記録する。submission SHA は inference port 後に記録する。
- Kaggle package bootstrap 確認方針: push 前に `prepare-kaggle-notebooks --strict` と generated metadata を確認する。

## GPU Cost Guard

- active variants: 1 (`exp183_selector_confidence_addonly`)
- LightGBM configs: 3
- folds: 5
- total boosters: 15
- control / parent retraining: なし
- train notebook split: なし。単一 GPU train notebook で実行する。

## リスク

- リークリスク: exp183 output に評価列が含まれるため、downstream feature builder で明示除外する。
- CV/LB 不一致リスク: exp160 は CV positive / LB negative だったため、global OOF 改善だけでは submit しない。
- ランタイム/メモリリスク: exp148 + exp183 selected-path features は full-row 3,783,989 rows。GPU train 15 boosters は exp148/exp160 と同等。
- 再現性リスク: exp183 selected path は upstream train-side artifact であり、raw-test current-test feature parity は未確認。submit には別途 inference port が必要。
