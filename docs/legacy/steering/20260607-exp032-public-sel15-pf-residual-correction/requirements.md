# 要件

## 依頼

`public_sel15_pf_residual_correction` を実装する。

## 制約

- Route: `pf_beam`
- 親 artifact は `exp029_public_sel15_pf_oof_feature_generation/features/public_sel15_pf_oof_features.csv.gz`。
- 目的変数は `target_tvt - pf_pred` とし、最終予測は `pf_pred + residual_model_pred`。
- `exp026_oof` は exp029 artifact で全欠損のため今回の特徴量から除外する。
- `target_tvt`、`pf_error`、`last_anchor_error`、`beam_error` は scoring / target 以外の特徴量に使わない。
- same-OOF 的な当てはまりでは採用判断せず、original-fold OOF と well-hash holdout の両方を見る。

## 受け入れ基準

- `exp032_public_sel15_pf_residual_correction` が作成され、`config.yaml` の route / lineage / leakage policy が明記されている。
- train notebook が、入力確認、残差補正 audit、metrics/artifact 保存をセル単位で追える。
- residual correction audit script が、original-fold OOF と well-hash holdout の両方で候補を評価する。
- required controls は少なくとも `public_pf_selector` と `pf090_hold010` を含む。
- `task validate-exp EXP=exp032_public_sel15_pf_residual_correction` と script の lint が通る。
