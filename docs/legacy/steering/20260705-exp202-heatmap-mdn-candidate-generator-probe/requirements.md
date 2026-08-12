# 要件

## 依頼

`KAGGLE_DIRECTION.md` の backlog `heatmap_mdn_candidate_generator_probe` を実験化する。exp182/184 の heatmap MTP 系を、PF/Beam の direct replacement ではなく候補生成器として再設計し、既存 PF/Beam/likPF/sc/hyb 候補 union に新しい heatmap topK TVT/path 候補を足す価値があるかを train-side GPU probe で確認する。

## 制約

- Route: `pf_beam`
- 親: `exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe`、`exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158`、`exp099_pf_multi_observation_likelihood_probe`
- 公開 discussion 699853 / example notebook の `K=10` path/logit head と closest-mode loss を参照する。
- 入力は exp182 と同じ 5ch heatmap (`typewell GR`, `horizontal GR`, 差分, observed `TVT_input` history SDF, observed mask) を初手に固定する。
- 初回は `candidate_real_w128_b64_fullfold` 1 spec x 5 folds = 5 CNN models。geometry / shuffled / no-GR control は再学習せず、exp182 の保存済み結果を根拠として参照する。
- `pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb` を必須の既存候補 union として比較する。dense family は cache に存在すれば評価し、存在しない場合は missing candidate として記録する。
- direct TVT replacement、softmax weighted average、PF weight replacement、postprocess blend、inference port、submit は行わない。
- valid/test true TVT、oracle best、true-error rank、OOF absolute error を feature source や inference-time selection に漏らさない。
- 再現性: `docs/06_reproducibility.md` に従い、GPU 学習、candidate cache SHA、prediction SHA、decompressed gzip SHA を記録する。

## 受け入れ基準

- `docs/legacy/steering/20260705-exp202-heatmap-mdn-candidate-generator-probe/` に要件、設計、tasklist がある。
- `experiments/exp202_heatmap_mdn_candidate_generator_probe/` に config、train/inference notebook source、notebook、README、SESSION_NOTES、result、metrics placeholder がある。
- train notebook は、active run spec、CNN model 数、候補 cache、既存候補列、禁止する target-derived columns、保存する生成物を表示する。
- heatmap topK 候補を `*_heatmap_candidates.csv.gz` として保存する。
- plot / readout 用に、validation sample ごとの deduplicate 済み center-row top10 candidate に対応する local 128-row path を npz + index CSV として保存する。これは full-well trajectory stitch ではなく、学習、selector、推論、提出には使わない。
- 既存候補 union と `existing_plus_heatmap_topK` の oracle RMSE / within10 / new-best rate を overall、well、distance bucket で保存する。
- Kaggle push 前に、実行予定が 1 active heatmap candidate-generator spec、5 folds、5 CNN models、0 LightGBM boosters、control / parent 再学習なしであることを `SESSION_NOTES.md` に記録する。
- deterministic anchor とは扱わず、GPU stochastic の範囲を明記する。
