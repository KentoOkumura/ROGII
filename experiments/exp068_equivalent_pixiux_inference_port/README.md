# exp068_equivalent_pixiux_inference_port

## 状態

- ルート: ml_model
- 状態: discarded
- 親実験: `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`
- 参照 branch: `exp039_ravaghi_single_lgbm_inference_submit`
- 作成日: 2026-06-13

## 仮説

元のバックログどおり、`exp039 型 branch` の価値を `exp063` 上で再評価する。train 側では exp039/exp038 系の CV surface に exp063 の tracker/PF/Beam output features を `id` で join し、exp063 の Pixiux LightGBM config family を同じ CV で再学習評価する。

## 範囲

PF/Beam 特徴量の再生成は行わない。`exp063` が保存した tracker/PF/Beam train features を使用する。`exp063` の実装ファイルは変更せず、exp068 側で exp039 CV 評価を追加する。

LightGBM は GPU 実行を使用しつつ、`deterministic=true` / `force_col_wise=true` を設定する。

## 検証方針

Kaggle train notebook では `leave_one_original_fold_out` と `well_hash_holdout` を実行し、exp063 LightGBM 3 configs と `lgb_mean` の RMSE、well 別 error、OOF predictions を保存する。加えて primary audit の best iteration を使い、全 joined rows で full LightGBM boosters を保存する。

Kaggle inference notebook は exp068 train output の full boosters を読み、hidden test 上で exp063 replay features を生成して `submission.csv` を作る。静的な exp063 inference prediction artifact は使わない。

## 所見

初回 Kaggle train v4 では CV 評価のみ完了。レビュー後、再学習 full model artifact を保存して inference に使う設計へ修正したが、2026-06-16 のユーザー指示により exp068 は破棄した。invalid submission ref `53654439` は採用しない。

## 次のアクション

なし。代替 backlog `exp073_exp039_cv_reassessment` で、対象だけを exp063 から exp073 に差し替えて exp039 CV 評価を行う。

## 参照ファイル

- 設定: `config.yaml`
- セッションノート: `SESSION_NOTES.md`
- 結果: `result.md`
- メトリクス: `metrics.json`
- 学習 notebook: `exp068_equivalent_pixiux_inference_port_train.ipynb`
- 推論 notebook: `exp068_equivalent_pixiux_inference_port_inference.ipynb`
