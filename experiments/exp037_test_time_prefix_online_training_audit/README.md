# exp037_test_time_prefix_online_training_audit

## 状態

- ルート: `ml_model`
- 状態: 完了・fold外選択FAIL・不採用
- CV: 12.870780（選択した比較基準）
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-08
- 親実験: `exp026_pseudo_tail_bucket_shrink_inference_submit`

## 仮説

見えている `TVT_input` prefix から作った小重みのonline training rowsを追加すると、
exp026の固定bucket-shrink比較基準をfold外でも改善できるか監査する。

## 検証方針

- 同一OOF best: `online_weight_0_20`、12.844383。
- leave-one-original-fold-out選択: 12.999364。
- well-hash holdout選択: 12.970333。
- 比較基準: 12.870780。

fold外の2評価で比較基準を悪化させたため、online trainingを推論へ移植しない。

## 所見

同一OOFの小改善は確認したがfold固有の適応と判断し、不採用で完了した。
選択手法は`exp026_bucket_shrink_control`を維持する。

## 参照ファイル

- 設定: `config.yaml`
- セッションノート: `SESSION_NOTES.md`
- 結果: `result.md`
- メトリクス: `metrics.json`
- 学習 notebook: `exp037_test_time_prefix_online_training_audit_train.ipynb`
- 推論 notebook: `exp037_test_time_prefix_online_training_audit_inference.ipynb`

## 読み方

この README は実験フォルダの入口です。仮説、変更点、実行コマンド、出力、失敗理由、次のアクションは `SESSION_NOTES.md` と `result.md` を正とします。

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録します。
