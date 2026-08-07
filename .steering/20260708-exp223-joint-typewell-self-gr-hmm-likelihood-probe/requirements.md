# 要件

## 依頼

`joint_typewell_self_gr_hmm_likelihood_probe` backlog を `exp223_joint_typewell_self_gr_hmm_likelihood_probe` として実装する。`exp209` exact HMM の typewell GR emission を主軸に残し、visible prefix 由来 self-GR motif likelihood を弱い clipped boost として同時利用する。

## 制約

- Route: `ensemble`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 新規 LightGBM 学習はしない。
- exp072 full replay cache は再生成せず、保存済み exp072 cache を比較基準として読む。
- raw-test inference、submission、Kaggle submit は初期実装範囲外。
- self-GR surface は raw GR と finite `TVT_input` prefix だけから作る。
- unknown-suffix train true TVT、OOF absolute error、oracle best、true-error rank を self-GR search、quality、alpha、clip、normalization に使わない。
- self-GR 由来の candidate TVT、hard replacement、hard switch、softmax average、row-wise gate は実装しない。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理なし、HMM no RNG、gzip decompressed SHA 記録方針を設計に明記する。

## 受け入れ基準

- `experiments/exp223_joint_typewell_self_gr_hmm_likelihood_probe/` に config、train/inference notebook source、HMM helper、comparison helper、記録ファイルが揃っている。
- 初回 active variants は runtime 制限を優先し、`alpha=[0.07, 0.15]`、`clip=[1.0]`、`mode=[boost_only]` の 2 通り。
- `alpha=0.03` や `symmetric` mode は、初回 2 variant run が安定してから追加候補として扱う。
- train notebook が cost guard として variant 数、LightGBM config 数 0、fold 数 0、booster 数 0、control retraining なしを表示する。
- 全 well を対象にできる実装で、`feature_cache.hmm.max_wells` が `null` の場合は 773 wells 全体を走査する。
- comparison が overall、distance bucket、hidden-like、by-well、HMM std、step-delta、self-GR quality / agreement bucket を出力する。
- Jupytext conversion、構文チェック、`ruff --select F821`、`validate_experiment.py` が通る。
- deterministic anchor としては扱わず、Kaggle train 後は feature content SHA と kernel version を記録する。
