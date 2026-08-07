# 要件

## 依頼

`exp072_pf_z_seedbag_scale_cache` 相当の仮説を、占有済みの `exp072` 接頭辞ではなく最新の空き実験番号で実装する。実験名は `exp104_pf_z_seedbag_scale_cache` とし、pf_z seedbag / scale cache を生成して既存 PF/Beam 候補と比較する。

## 制約

- Route: `pf_beam`
- 親実験: `exp100_pf_z_unified_velocity_observation_prior`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- 比較対象: exp072 cache の `pf_z`、`likpf_mean`、存在する `likpf_scale_*`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic PF seedbag は well 単位の stable seed とし、gzip 生成物は decompressed content SHA を記録する。
- 初回 full run は Kaggle Notebook 上で実行する。ローカルには exp072 cache 本体がないため full 比較をローカルで代替しない。
- この実験は train-side diagnostic。submission / inference candidate は選ばない。

## 受け入れ基準

- `experiments/exp104_pf_z_seedbag_scale_cache/` に config、train/inference notebook、補助 `.py`、README、SESSION_NOTES、result、metrics がある。
- `config.yaml` の `experiment.route` が `pf_beam` で、lineage と leakage policy が明記されている。
- train notebook が exp072 cache を読み、`pf_z_seedbag_mean` / `pf_z_seedbag_scale_3/5/8/12` を生成し、exp072 `pf_z` / `likpf_mean` と同じ candidate metrics で比較する。
- 生成物として candidate metrics、bucket metrics、by-well metrics、seedbag quality、candidate wide/long cache、summary JSON を保存する。
- deterministic anchor として扱わず、Kaggle kernel version、input cache SHA、生成物 SHA を記録対象にする。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
