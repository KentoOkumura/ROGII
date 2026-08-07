# exp025_pseudo_tail_postprocess_cv_audit セッションノート

## 目的

バックログ先頭の `pseudo_tail_postprocess_cv_audit` を実装する。`exp024` raw pseudo-tail anchor を壊さないため、postprocess は OOF same-fit ではなく original fold 外 / well-hash holdout の監査で改善が残る場合だけ inference に進める。

## 現在の状態

- 状態: 完了
- 親実験: `exp024_pseudo_tail_inference_postprocess`
- OOF 再現元: `exp023_pseudo_tail_distance_augmentation`
- 対象 variant: `pseudo_tail_3_cutoffs_distance_balanced`
- 親 clean CV: 12.942938
- 親 Public LB: 12.166
- selected method: `exp014_bucket_shrink_params`
- CV: 12.870780

## コマンドログ

- 2026-06-06: `uv run python scripts/new_steering.py --experiment exp025_pseudo_tail_postprocess_cv_audit` で steering docs を作成。
- 2026-06-06: `uv run python scripts/new_experiment.py --name exp025_pseudo_tail_postprocess_cv_audit --source experiments/exp023_pseudo_tail_distance_augmentation` で exp023 から実験を作成。
- 2026-06-06: notebook 名、`settings.py`、`config.yaml`、README を exp025 用に更新。
- 2026-06-06: `pseudo_tail_postprocess_cv_audit.py` を追加。exp023 selected recipe の fold-safe OOF を再生成し、fixed candidates と bucket alpha fit を original-fold / well-hash holdout で監査する。
- 2026-06-06: `kaggle kernels push -p experiments/exp025_pseudo_tail_postprocess_cv_audit/kaggle/train` で train version 1 を push。Kaggle URL slug は `kentookumura/exp025-pseudo-tail-postprocess-audit-train`。
- 2026-06-06: `kaggle kernels status kentookumura/exp025-pseudo-tail-postprocess-audit-train` は Kaggle API 500 を返したが、`kaggle kernels output kentookumura/exp025-pseudo-tail-postprocess-audit-train -p /tmp/kaggle-output/exp025_pseudo_tail_postprocess_cv_audit/train` で output を取得。
- 2026-06-06: Kaggle full CV は raw pseudo-tail 12.942938 を再現し、fixed `exp014_bucket_shrink_params` が 12.870780、original-fold bucket alpha fit が 12.887830、well-hash bucket alpha fit が 12.879401。fixed-candidate selection は original-fold / well-hash の全 holdout で `exp014_bucket_shrink_params` を選んだ。

## 変更点

- `exp023` は row-level OOF を保存していないため、exp025 train notebook で selected pseudo-tail recipe を再学習して OOF 集計を作る。
- 常設 artifact は大きな row OOF ではなく、postprocess metrics、selection rows、alpha rows、bucket summary、fold metrics、source summary、feature importance に限定する。
- fixed candidates は raw、exp014 bucket shrink、near-only shrink、far alpha、near shrink + far alpha を比較する。
- same-OOF bucket alpha fit は診断値として保存し、採用判定には使わない。

## Artifacts

- `artifacts/pseudo_tail_postprocess_metrics.csv`
- `artifacts/pseudo_tail_postprocess_selection.csv`
- `artifacts/pseudo_tail_postprocess_alphas.csv`
- `artifacts/pseudo_tail_postprocess_bucket_summary.csv`
- `artifacts/pseudo_tail_postprocess_fold_metrics.csv`
- `artifacts/pseudo_tail_source_summary.csv`
- `artifacts/pseudo_tail_feature_importance.csv`
- `artifacts/pseudo_tail_postprocess_summary.json`
- `metrics.json`

## 結果

- Raw pseudo-tail: 12.942938
- `exp014_bucket_shrink_params`: 12.870780
- `far_alpha_110`: 12.881263
- `far_alpha_105`: 12.906041
- `near_shrink_far_105`: 12.908677
- `near_only_shrink`: 12.945566
- same-OOF bucket alpha fit: 12.863570, 診断値のみ
- leave-one-original-fold-out bucket alpha fit: 12.887830
- well-hash holdout bucket alpha fit: 12.879401

固定候補 selection は original-fold / well-hash の全 holdout で `exp014_bucket_shrink_params` を選び、どちらも 12.870780。raw から -0.072158 改善した。

## 次のアクション

1. `exp026_pseudo_tail_bucket_shrink_inference_submit` を切り、exp024 raw pseudo-tail inference flow に `exp014_bucket_shrink_params` を適用する。
2. Kaggle inference output を submit-check し、提出する。
3. Public LB と CV の一貫性を記録する。
