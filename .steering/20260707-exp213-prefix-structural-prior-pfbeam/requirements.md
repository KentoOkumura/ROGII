# 要件

## 依頼

KAGGLE_DIRECTION backlog `prefix_structural_prior_pfbeam` を実装する。known prefix の `TVT_input + Z` を structural surface state として使い、PF/Beam の初期状態と transition prior を作る。

## 制約

- Route: `pf_beam`
- 観測 likelihood は raw GR を主比較に固定し、P0-A の affine calibrated GR とは分けて効果を読む。
- `TVT_input + Z` fit は known prefix のみを使う。evaluation tail の true TVT、target、oracle error は fit や path selection に使わない。
- hard window、candidate invalidation、true-error tuned strength は使わない。
- top-K path、cost gap、path spread、PF ESS/resampling、path jump rate を保存し、mode collapse を確認できるようにする。
- 再現性: `docs/06_reproducibility.md` に従い、PF/Beam seed policy、gzip decompressed SHA、Kaggle bootstrap の扱いを記録する。
- Kaggle GPU / LightGBM 学習 / booster 生成は含めない。

## 受け入れ基準

- `experiments/exp213_prefix_structural_prior_pfbeam/` が標準構造で作成される。
- `config.yaml` の `experiment.route` が `pf_beam` で、active variant 数、PF/Beam config、fold/booster 数が記録される。
- raw classic baseline と raw structural variants が同一 target wells / PF seeds / particles / Beam 幅で比較される。
- PF は structural fit から初期速度と weak velocity pull を使う。
- Beam は last-known start を維持しつつ、absolute TVT soft prior と step-delta cost を使う。
- candidate metrics、filter delta、bucket/group/by-well、PF diagnostics、top-K Beam candidates、summary JSON を保存できる。
- deterministic anchor として扱わない場合も、Kaggle train 実行後は input / row candidates の decompressed content SHA を記録する。
