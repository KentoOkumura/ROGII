# 設計

## アプローチ

exp157 の supervised PF/Beam/dense candidate ranker をベースに、exp181 で有効性が見えた cluster-outlier prior signal を add-only feature として追加する。学習した candidate error ranker の OOF predicted-error surface を exp158 と同じ well-local Viterbi grid に渡し、row-wise selector と continuity selector の両方を評価する。

## 実験範囲

- 対象実験: `exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector`
- Route: `pf_beam`
- 親実験: `exp158_segment_continuity_selector_on_exp157`
- 変更する変数: candidate selector の score feature set
- 固定する変数: 8候補、GroupKFold by well、exp157 dense enrichment、exp158 Viterbi grid、direct correction なし
- 比較対象: `likpf_mean` 11.594897672、exp157 row-wise 10.795799837、exp158 continuity 10.789163253

## Feature 設計

- row feature: `own_cluster_dist_z`、`nearest_other_closer`、`nearby_majority_diff_k8`、`any_outlier_signal_k8`、gate の well 内 ratio
- prior feature: typewell native overlap 1 / 0.999、spatial xy+trajectory k8、spatial xy-only k8 の prior TVT / std / count / neighbor
- candidate-long feature: `prior - candidate`、abs/norm delta、valid prior、std/count/neighbor、gate x candidate family、alpha 0.2 clip 20/40 の correction magnitude と clip hit
- result-only subgroup: exp115 spatial/typewell-purged valid role

## 再現性設計

- seed policy: GroupKFold seed 42、LightGBM fold seed、candidate-long row subsample seed を固定する。
- stochastic 処理の有無: exp183 自体は LightGBM histogram training と fixed subsample が stochastic component。PF/Beam 生成は再実行しない。
- PF/Beam / likelihood-PF / seed bagging の有無: upstream exp099 / exp072 / exp109 / exp114 output を固定入力として読む。
- 並列処理と乱数の関係: feature generation は RNG を使わない。candidate-long subsample は fold seed の local RNG のみ。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU disabled。
- train cache / test feature regeneration の SHA 記録方針: gzip は decompressed SHA を summary に記録する。
- model manifest / prediction / submission SHA 記録方針: model manifest と OOF prediction SHA を記録する。submission は作らない。
- Kaggle package bootstrap 確認方針: prepare 後に train/inference metadata と bootstrap support files を確認する。

## リスク

- リークリスク: cluster/prior source に true TVT や oracle rank を入れない。exp115 role は result-only。
- CV/LB 不一致リスク: train-side selector なので positive でも raw-test parity / hidden-like stress を見ずに submit しない。
- ランタイム/メモリリスク: 3 configs x 5 folds = 15 boosters。long model は fold ごと最大 650k rows に subsample。
- 再現性リスク: upstream PF/Beam/prior generation は固定 Kaggle output として扱うため、deterministic submission anchor ではない。
