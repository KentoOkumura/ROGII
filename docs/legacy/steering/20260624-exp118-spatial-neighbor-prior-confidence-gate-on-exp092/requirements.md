# 要件

## 依頼

`spatial_neighbor_prior_confidence_gate_on_exp092` を実装し、exp114 の spatial neighbor prior を exp092 に対して直接補正候補として使える row / well があるか train-side に診断する。

## 制約

- Route: `ml_model`
- 提出物生成ではなく、exp092 OOF 予測に対する posthoc confidence / gate audit とする。
- exp114 の prior は固定済み OOF 生成物を入力にし、valid fold の true TVT を prior 生成や gate feature には使わない。
- exp092 の OOF 予測を固定入力にし、新しい LightGBM 学習や PF/Beam 再生成はしない。
- 再現性: `docs/06_reproducibility.md` に従い、入力 gzip は decompressed content SHA を記録する。
- global OOF が改善しても、worst-well regression、bucket、path continuity が悪ければ inference port / submit に進めない。

## 受け入れ基準

- `experiments/exp118_spatial_neighbor_prior_confidence_gate_on_exp092/` に設定、train notebook、no-submission inference notebook、実装スクリプトがある。
- train notebook は exp114 OOF prior と exp092 OOF prediction の存在確認、gate audit 実行、生成物確認をセル単位で追える。
- 生成予定物として gate metrics、by-well delta、bucket metrics、path continuity、上位 gated prediction、summary JSON が定義されている。
- `make validate-exp EXP=exp118_spatial_neighbor_prior_confidence_gate_on_exp092` が通る。
- Kaggle train 実行後は `SESSION_NOTES.md`、`result.md`、`metrics.json` に CV / 解釈 / 次アクションを追記する。
