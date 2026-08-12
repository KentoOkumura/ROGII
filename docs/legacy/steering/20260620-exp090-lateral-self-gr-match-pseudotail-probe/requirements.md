# 要件

## 依頼

`lateral_self_gr_match_pseudotail_probe` を実装する。

## 制約

- Route: `ml_model`
- 親実験: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- 再現性: `docs/06_reproducibility.md` に従い、Kaggle bootstrap、GPU LightGBM、feature schema、model/prediction SHA の記録方針を設計に明記する。
- 評価区間の true `TVT` は GR match feature generation に使わない。
- GR match は同一 horizontal well 内の target-free summary feature に限定し、直接 TVT 置換や typewell 外部照合にはしない。
- exp008 / exp017 / exp042 の GR alignment 悪化を既知リスクとして扱い、near rows と worst-well の悪化を確認する。

## 受け入れ基準

- `experiments/exp090_lateral_self_gr_match_pseudotail_probe/` に config、train/inference notebook、補助 `.py`、記録ファイルが揃っている。
- `config.yaml` の `experiment.route` が `ml_model` で、lineage と leakage policy が明記されている。
- train notebook が exp072 cache 確認、self-GR feature ablation 実行、metrics/feature importance 確認をセル単位で追える。
- 補助 `.py` が exp073 196 feature control と self-GR variants を同一 GroupKFold 条件で比較できる。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
