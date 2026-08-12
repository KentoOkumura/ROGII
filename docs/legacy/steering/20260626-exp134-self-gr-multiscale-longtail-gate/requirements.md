# 要件

## 依頼

KAGGLE_DIRECTION の backlog `self_gr_multiscale_longtail_gate` を exp134 として実装する。
exp090 の self-GR multiscale signal を、直接 TVT 候補や submit candidate ではなく、
high-drift / PF-dense disagreement gate の補助 confidence として評価する。

## 制約

- Route: `ml_model`
- 新規 LightGBM 学習はしない。保存済み/再生成した train-side signal の posthoc audit とする。
- self-GR 由来値を単独 TVT 予測、hard replacement、submission candidate として扱わない。
- 評価対象は exp092 / exp073 / PF/Beam failure map と整合する pseudo-tail train rows。
- valid/test の true TVT を gate 条件や feature source に使わない。true TVT は評価、bucket、oracle readout のみに使う。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

- `experiments/exp134_self_gr_multiscale_longtail_gate/` に config、train/inference notebook、補助 script、記録ファイルがある。
- train notebook で入力 artifact の存在確認、audit 実行、metrics / summary 保存が追える。
- `self_gr_sc25_delta_tvt`、`self_gr_sc25_score`、`self_gr_sc25_l2`、distance bucket、GR missingness、scale disagreement を含む gate signal を評価できる。
- overall、common worst proxy、distance/tail bucket、near rows、PF-dense disagreement bucket、by-well regression を保存する。
- 結論は「submit 目的なし」「LightGBM 学習なし」「後続 add-only / ranker ablation に渡すか棄却するか」を明示する。
- deterministic anchor としては扱わない。gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
