# 要件

## 依頼

`compact_rank_slot_features_on_exp098` を実装する。

exp098 の rank-slot structured features は exp073 / exp077 を改善したが、追加 64 列には重複・低 utility 列が含まれる。exp098 を親にして、より小さい rank-slot feature set を同じ exp073/exp072 surface 上で評価する。

## 制約

- Route: `ml_model`
- 親実験は `exp098_selector_rank_slot_features_on_exp073`。
- base feature surface、target、GroupKFold by well、LightGBM family は exp098 と同じにする。
- PF/Beam 候補を直接 selector / soft average / postprocess replacement として使わない。
- rank slot は target-free score だけで作り、評価区間 true TVT は rank 生成に使わない。
- Kaggle Notebook 実行を正とし、ローカル notebook 実行はしない。
- 再現性: `docs/06_reproducibility.md` に従い、upstream cache SHA、model manifest、prediction SHA を Kaggle train 完了後に記録する。

## 受け入れ基準

- exp105 の `config.yaml`、train notebook、補助 `.py`、`SESSION_NOTES.md`、`result.md`、`metrics.json` が compact 実験内容と一致している。
- active variant は `compact_rank_slot_features` のみ。
- compact group は base 196 features に 22 rank-slot features を追加する。
- 削る列は `rank*_u_fit_degree`、pairwise candidate delta、rank 間 `u_diff` / `u_absdiff`、`u_corr` / `u_resid` の符号反転ペア、source flags とする。
- `make validate-exp EXP=exp105_compact_rank_slot_features_on_exp098` が通る。
- Kaggle push 前の train package 生成が strict mode で通る。
