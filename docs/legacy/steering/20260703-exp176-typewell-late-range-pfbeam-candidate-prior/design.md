# 設計

## アプローチ

exp157 の supervised candidate ranker を親にし、8 候補 (`pf_ancc`, `beam_mean`, `likpf_mean`, `sc_ens`, `hyb`, `tvt_dense`, `tvt_densew`, `tvt_dense50`) は固定する。

raw train の typewell TVT range と visible prefix の last known TVT から、well ごとの `typewell_min`、`typewell_max`、`typewell_span`、`known_last_pct` を作る。各候補に対して `candidate_pct = (candidate_tvt - typewell_min) / typewell_span` を計算し、late prefix なのに前半 range に戻る候補を低信頼にする特徴量を追加する。

特徴量は row-level summary と candidate-long feature の両方に入れる。候補を除外せず、LightGBM ranker が無視できる形にする。

## 実験範囲

- 対象実験: `exp176_typewell_late_range_pfbeam_candidate_prior`
- Route: `ensemble`
- 親実験: `exp157_candidate_ranker_feature_enrichment`
- 変更する変数: `candidate_pct`、`candidate_pct_minus_known_last_pct`、fixed lower-bound flag、`known_last_pct - margin` flag、late-prefix interaction
- 固定する変数: candidate set、exp099 / exp072 input cache、GroupKFold by well、LightGBM 3 family、5 folds、long-frame sampling policy

## 再現性設計

- seed policy: exp157 と同じ `fixed_groupkfold_and_lightgbm_seed`
- stochastic 処理の有無: LightGBM training と candidate-long training row subsample は stochastic。feature generation は deterministic。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。保存済み exp099 / exp072 cache を読む。
- 並列処理と乱数の関係: feature generation に global RNG は使わない。long-frame subsample は fold seed の local RNG。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、`enable_gpu=false`。
- train cache / test feature regeneration の SHA 記録方針: exp099 / exp072 gzip は decompressed content SHA を記録する。
- model manifest / prediction / submission SHA 記録方針: model manifest、OOF prediction、feature schema、metrics SHA を記録する。submission は作らない。
- Kaggle package bootstrap 確認方針: prepare 後に generated notebook metadata と bootstrap 内 config を確認する。

## リスク

- リークリスク: true TVT / oracle label / true error rank を feature に使わない。label は supervised train-side OOF の目的変数に限定する。
- CV/LB 不一致リスク: exp157 / exp158 は train-side supported だが row-wise / segment selector の hidden generalization は未確認。positive でも raw-test parity と hidden-like stress を必須にする。
- ランタイム/メモリリスク: exp157 と同じ 15 boosters で CPU runtime は長い。追加 feature で long-frame が広がるため、`max_train_rows_per_fold=650000` を維持する。
- 再現性リスク: upstream PF/Beam cache は既存生成物に依存する。exp176 自身は deterministic submission anchor ではない。
