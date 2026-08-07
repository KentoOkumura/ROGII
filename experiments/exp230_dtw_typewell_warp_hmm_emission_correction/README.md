# exp230_dtw_typewell_warp_hmm_emission_correction

## 状態

- 完了・不採用。Kaggle train v2 complete
- Route: `pf_beam`
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- CPU-only train-side diagnostic。推論、提出、raw-test regeneration は対象外。

## 仮説

full horizontal GR と typewell GR の constrained DTW / elastic registration は、直接 TVT path としては危険だが、exp209 exact HMM の raw typewell GR emission に小さく足す補助 emission なら、HMM の worst tail や hidden-like subgroup を改善できる可能性がある。

## 検証方針

- exp209 HMM transition / raw GR emission / grid は固定。
- DTW 補助 emission は `alpha=0.05` と `alpha=0.10` の 2 variants。
- exp072 full replay は再生成せず、保存済み cache を comparison baseline として読む。
- overall、distance bucket、exp115 hidden-like、by-well regression、HMM std、step-delta を読む。

## 所見

Kaggle train v2 は `kentookumura/exp230-dtw-hmm-emission-train` として完了。best HMM は `hmm_dtw_a005_s1200` RMSE 13.611292322 で、exp072 `likpf_mean` 11.594897668 から +2.016395 悪化した。near buckets は小改善したが、`1000_plus` と exp115 hidden-like が悪化し、worst regression も大きいため raw-test regeneration、inference、submit は行わない。

## 参照ファイル

- 学習 notebook: `exp230_dtw_typewell_warp_hmm_emission_correction_train.ipynb`
- 推論 notebook: `exp230_dtw_typewell_warp_hmm_emission_correction_inference.ipynb`
- 設定: `config.yaml`
- 実装: `exact_hmm_smoother.py`
