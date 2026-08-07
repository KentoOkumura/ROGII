# exp048_ravaghi_single_model_feature_parity_revisit

## 状態

- ルート: `ml_model`
- 状態: implemented
- CV: pending
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-10
- 親実験: `exp043_ravaghi_feature_family_ablation_matrix`

## 仮説

Ravaghi 由来特徴を、同一生成条件・同一特徴列・同一欠損処理の単体モデルとして再評価する。
成功条件は弱い base geometry ではなく、`public_pf_selector` と `pf090_hold010` の直接比較基準を
両 holdout で上回ること。

## 検証方針

検証 split、リーク確認、Kaggle 実行条件は `config.yaml`、`SESSION_NOTES.md`、`result.md` を正とする。
raw、fixed bucket shrink、anchor gate、public PF blend は別候補として記録する。

## 所見

成功点、失敗点、次のアクションは `result.md` と `SESSION_NOTES.md` を参照する。

## 参照ファイル

- 設定: `config.yaml`
- セッションノート: `SESSION_NOTES.md`
- 結果: `result.md`
- メトリクス: `metrics.json`
- 学習 notebook: `exp048_ravaghi_single_model_feature_parity_revisit_train.ipynb`
- 推論 notebook: `exp048_ravaghi_single_model_feature_parity_revisit_inference.ipynb`

## 読み方

この README は実験フォルダの入口です。仮説、変更点、実行コマンド、出力、失敗理由、次のアクションは
`SESSION_NOTES.md` と `result.md` を正とします。
