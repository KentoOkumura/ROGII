# 要件

## 依頼

`tvt_dense_high_drift_confidence_gate_on_exp092` を実装する。第一段階は LightGBM の新規学習を行わず、既存の exp092 OOF prediction と exp072 PF/Beam/dense feature cache を使った train-side posthoc gate 評価に限定する。

## 制約

- Route: `ml_model`
- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- 比較対象: exp092 `lgb1`、exp073 `lgb_mean`、`likpf_mean`、`tvt_dense` 系候補。
- 再現性: `docs/06_reproducibility.md` に従い、入力 gzip は decompressed content SHA を記録する。
- LightGBM / CatBoost / ranker の新規学習は行わない。
- gate 条件には validation/test true TVT、oracle candidate、error label を使わない。
- `tvt_dense` 全体置換や broad row-wise switch を提出候補にしない。
- inference / submission はこの実装段階では作らない。

## 受け入れ基準

- exp135 実験フォルダが作成され、`config.yaml` に route、親実験、gate grid、再現性方針が明記されている。
- train notebook が Kaggle Notebook 実行で、入力確認、posthoc audit、生成物確認を追える構成になっている。
- `tvt_dense` / `tvt_densew` / `tvt_dense50` / `tvtF_ANCC` の single candidate と oracle headroom を評価できる。
- high-drift / high-disagreement regime の target-free well/segment gate を小 grid で比較できる。
- 評価には overall RMSE、within10、common PF+ML worst 26 wells、PF `likpf_mean` worst50、tail bucket、near-row、path continuity、raw-test parity checklist を含む。
- `result.md` / `SESSION_NOTES.md` / `metrics.json` が未実行状態として正しく記録されている。
