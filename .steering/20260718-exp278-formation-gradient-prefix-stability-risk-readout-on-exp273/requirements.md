# 要件

## 依頼

`KAGGLE_DIRECTION.md` の backlog
`formation_gradient_prefix_stability_risk_readout_on_exp273` を
`exp278_formation_gradient_prefix_stability_risk_readout_on_exp273` として実装する。
exp273 の direct 2D-gradient 候補で一部 well に生じた巨大回帰を、known prefix 内の
formation-plane 不安定性だけから事前識別できるかを 0-booster readout で監査する。

## 制約

- Route: `pf_beam`。新しい HMM path は生成せず、exp273 の PF/Beam 系候補生成を監査する。
- exp273 shard 0/1 candidate、aggregate plane diagnostics、by-well metrics を期待 SHA で固定する。
- raw train から読む値は同一 well の `MD/X/Y/Z/TVT_input` だけとし、`TVT`、evaluation-tail target、
  formation label、他 well target は plane feature に使わない。
- exp273 と同じ deterministic Huber plane を full / last-512 / last-256 known-prefix にだけ再計算する。
  generation guardのvalidity/fallback/zero-gradientは維持しつつ、min-points/rank-2を満たすwindowは
  diagnostic-only fitも保存し、guard不通過を理由に角度・大きさ・fit RMSEを消さない。
- plane fit 後に gradient 角度差、大きさ比、fit RMSE、rank ratio、condition number、validity flip を
  事前固定式で一つの stability risk へ集約する。outcome を見た重み・clip・window・閾値変更は禁止する。
- exp273 の true TVT / error / oracle は risk feature、outer fold、quantile の fit に使わず、
  risk 凍結後の readout outcome にだけ接続する。
- HMM 再実行、gradient scale / geometry guard 調整、hard gate、selector 学習、raw-test inference、
  submission は禁止する。
- 実行量は 0 variant / 0 model config / 0 trained fold / 0 booster、Kaggle CPU、GPU/internet offとする。
- 再現性は `docs/06_reproducibility.md` に従い、gzip は raw SHA と decompressed content SHA を分ける。

## 受け入れ基準

- full-prefix plane の rows、validity、fallback reason、gradient、geometry、fit RMSE が保存済み
  exp273 plane diagnostics と事前 tolerance 内で parity する。
- shard candidate から再集約した 5 gradient candidate の by-well RMSE が保存済み aggregate
  by-well metrics と parity する。
- 773 wells と full-valid 111 wells を欠損・重複なく読み、stable SHA256 で 5 audit outer folds に割り当てる。
- primary cohort は exp273 full-gradient-valid wells、primary outcome は 5 gradient candidate の
  well-level `delta_rmse_vs_scalar` 平均とし、stability risk との Spearman 方向を pooled と 5 foldsで保存する。
- fold 別正方向 5/5、pooled 正方向、最高 risk quintile の平均回帰が最低 quintile より大きい場合だけ
  「別実験の gate 設計根拠あり」と判定する。candidate 別・secondary outcome は primary guardを救済しない。
- plane diagnostics、stability features、candidate outcome、fold/pooled correlation、risk quantile、plot、
  summary、reproducibility manifest を保存する。
- notebook は self-contained な Jupytext percent source から生成し、入力、契約、plane 再計算、
  outcome 接続、guard、生成物をセル上で追える。
- `py_compile`、Ruff F821/E9、targeted tests、Jupytext round-trip、`validate-exp` が通る。
- 初回 full readout は Kaggle CPU を正とし、実装依頼だけでは push しない。
