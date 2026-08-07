# 要件

## 依頼

`segment_level_dense_candidate_verifier` の backlog を現 ML route submitted anchor の exp148 に更新した上で、LightGBM を新規学習しない posthoc segment verifier audit として実装する。

## 制約

- Route: `ml_model`
- 主比較対象は exp148 `lgb_mean` CV 8.501281182 / Public LB 7.960 とする。
- exp092 は旧 submitted anchor / historical baseline として保持する。
- LightGBM の新規学習、control 再学習、GPU 学習は行わない。
- `target_tvt`、true error、oracle best は scoring と readout のみに使い、verifier 条件には使わない。
- `tvt_dense` の全体置換、dense-only submission、row-wise hard switch はしない。
- 再現性: `docs/06_reproducibility.md` に従い、保存済み OOF / feature cache の SHA を記録する。

## 受け入れ基準

- `KAGGLE_DIRECTION.md` の backlog が exp148 baseline 前提に更新されている。
- exp154 実験フォルダ、config、train/inference notebook、補助 `.py` が exp148 base に対応している。
- train audit は no-new-LGBM で、planned boosters が 0 と記録されている。
- verifier は target-free feature のみで segment mask を作る。
- 評価出力に overall、PF worst50、common PF+ML worst26、exp148 worst50、near-row、path continuity、worst-well regression、raw-test parity が含まれる。
- deterministic anchor として扱わないことが記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録する。
