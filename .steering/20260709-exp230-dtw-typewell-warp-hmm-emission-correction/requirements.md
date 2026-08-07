# 要件

## 依頼

`dtw_typewell_warp_hmm_emission_correction` を実装する。親実装はユーザー指定により `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation` とする。

## 制約

- Route: `pf_beam`
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- DTW は full horizontal GR と typewell GR の constrained warp を使うが、直接 TVT candidate として採用しない。
- HMM の raw typewell GR emission と transition は exp209 の設定を維持し、DTW は小さい `alpha` の補助 emission に限定する。
- exp072 full replay / parent / control は再生成しない。保存済み exp072 cache を comparison baseline として読む。
- LightGBM config 数 0、fold 数 0、booster 数 0。GPU 学習なし。
- 真の tail TVT、oracle error、LB score、同一 OOF error による row-wise weight tuning は使わない。
- 再現性: `docs/06_reproducibility.md` に従い、DTW jitter は well 固定 SHA seed で記録する。

## 受け入れ基準

- `experiments/exp230_dtw_typewell_warp_hmm_emission_correction/` が exp209 親として作成されている。
- train notebook が DTW 補助 emission HMM feature cache と comparison metrics を生成できる。
- overall、distance bucket、exp115 hidden-like、by-well worst regression、step-delta、HMM std calibration が出力される。
- `config.yaml` に route、lineage、DTW variants、runtime、禁止事項が明記されている。
- `py_compile`、`ruff --select F821`、`validate-exp`、Jupytext 変換確認が通る。
