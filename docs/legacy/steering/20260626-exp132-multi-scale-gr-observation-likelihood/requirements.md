# 要件

## 依頼

`multi_scale_gr_observation_likelihood` バックログを `exp132_multi_scale_gr_observation_likelihood` として実装する。exp099 の raw multi-observation likelihood を直接 scorer として使う方向は崩壊したため、複数 window / 複数 offset / derivative / smoothed GR / local z-score / energy / decoy gap を使い、PF/Beam 候補の target-free confidence / verifier feature を作る。

## 制約

- Route: `pf_beam`
- 親実験: `exp099_pf_multi_observation_likelihood_probe`
- cache 親: `exp072_exp063_full_replay_feature_cache`
- PF/Beam / likelihood-PF は再実行しない。既存候補値を読むだけにする。
- true TVT は scoring / oracle / rank metrics / stress summaries のみに使う。
- direct top1、softmax、blend は診断候補として保存するが、成功条件にはしない。
- supervised ranker、LightGBM add-only、inference port、submission はこの実験の範囲外にする。
- 再現性: `docs/06_reproducibility.md` に従い、upstream cache SHA と gzip decompressed content SHA を記録する。

## 受け入れ基準

- `experiments/exp132_multi_scale_gr_observation_likelihood/` に config、補助コード、train / inference notebook、記録ファイルがある。
- `config.yaml` に route、lineage、leakage policy、multi-scale GR likelihood 設定、candidate sets、expected train artifacts が明記されている。
- train notebook から `multi_scale_gr_observation_likelihood.py` を実行し、candidate metrics、rank metrics、bucket metrics、by-well metrics、row context、candidate long、wide feature cache、summary JSON を保存できる。
- multi-scale GR likelihood は raw horizontal GR、row index、finite prefix TVT_input、既存候補 TVT だけから計算する。
- deterministic anchor として扱わない。model / prediction / submission SHA は対象外であることを明記する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
