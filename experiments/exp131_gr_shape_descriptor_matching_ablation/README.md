# exp131_gr_shape_descriptor_matching_ablation

## 状態

- ルート: pf_beam
- 状態: completed_train_side_audit
- CV: candidate likelihood audit only
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-26
- 親実験: `gr_shape_descriptor_matching_ablation` backlog

## 仮説

単点 GR 差、NCC、DTW/DWT add-only は既存実験で弱かったが、PF/Beam/likelihood-PF 候補集合には oracle headroom が残っている。raw GR 点一致ではなく、local z-score shape、derivative、curvature、energy、peak/trough proxy、missing gap を含む shape descriptor cost にすれば、既存候補の当たり外れを target-free に説明できる可能性がある。

## 変更点

- exp072 deterministic full replay train cache を固定入力として読む。
- `pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb` を対象候補にする。
- 候補 TVT を validation true TVT ではなく prefix `TVT_input` 空間に写し、評価 row 周辺の horizontal GR window と比較する。
- `raw_point_real`、`ncc_window_real`、`banded_shift_real`、`shape_descriptor_real`、`combo_descriptor_real` を比較する。
- `combo_descriptor_shuffled` と `no_gr_constant` を negative control として必ず評価する。
- 直接 TVT 候補、LightGBM add-only 大量投入、推論 port、提出は行わない。

## 検証方針

- Fold: train-side pseudo-tail audit の既存 cache に従う
- Group: well
- 指標: candidate AUC/logloss、within10 probability、topK coverage、rank-score top1 RMSE、bucket / by-well stress
- Leakage Check: descriptor normalization、window center、candidate generation に validation true TVT を使わない

## 実行入口

- 学習 notebook: `exp131_gr_shape_descriptor_matching_ablation_train.ipynb`
- 推論 notebook: `exp131_gr_shape_descriptor_matching_ablation_inference.ipynb`
- Kaggle 準備: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp131_gr_shape_descriptor_matching_ablation --notebook train --kernel-id kentookumura/exp131-gr-shape-descriptor-train --title 'exp131 gr shape descriptor train' --run-on-push --strict`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は明示的な smoke debug のみに限定する。

## 期待する生成物

- `exp131_gr_shape_descriptor_matching_ablation_descriptor_scores_train_features.csv.gz`
- `exp131_gr_shape_descriptor_matching_ablation_descriptor_scores_feature_schema.csv`
- `exp131_gr_shape_descriptor_matching_ablation_candidate_metrics.csv`
- `exp131_gr_shape_descriptor_matching_ablation_score_variant_metrics.csv`
- `exp131_gr_shape_descriptor_matching_ablation_rank_metrics.csv`
- `exp131_gr_shape_descriptor_matching_ablation_bucket_metrics.csv`
- `exp131_gr_shape_descriptor_matching_ablation_by_well.csv`
- `exp131_gr_shape_descriptor_matching_ablation_summary.json`

## 結果

Kaggle train v1 完了。`combo_descriptor_real` は within10 label の AUC 0.659206 / logloss 0.653906 で最良。shuffled-GR AUC 0.570007、no-GR AUC 0.500000 を明確に上回った。一方、score top1 で直接候補を選ぶと RMSE 84.919128 / within10 0.560979 まで崩壊し、既存 `likpf_mean` RMSE 11.594897 / within10 0.772807 に大きく負ける。

## 採用判断

shape descriptor は likelihood / confidence feature 材料としては支持する。direct scorer、hard switch、direct TVT candidate、inference port、submit はしない。

## 所見

full DTW path を避け、fixed window / fixed offsets / banded local shift proxy に制限した設計は Kaggle CPU で完走した。ただし DataFrame fragmentation warning が大量に出ており、今後同じ特徴量を再生成する場合は列追加をまとめる実装に直すとよい。生成済み wide cache は Kaggle output 上に存在するが大きいため、ローカル取得は途中で止めている。
