# exp088_sequence_model_residual_diversity

## 状態

- ルート: `ml_model`
- 状態: `implemented_not_run`
- CV: Kaggle train 未実行
- Public LB: なし
- 作成日: 2026-06-20
- 親: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- 入力 cache: `exp072_exp063_full_replay_feature_cache`

## 仮説

軽量な GRU / TCN は LightGBM と異なる sequence inductive bias を持つため、
exp073 OOF anchor の残差誤差に対して、単体 RMSE が勝てなくても誤差相関の低い
多様性候補になる可能性がある。

既存 backlog では exp063 error map 後の候補として書かれていたが、
exp063 は現在の ML route anchor ではない。実装対象は exp073 deterministic full replay
OOF anchor に更新する。

## 検証方針

- exp073 `gpu_repro_guard_dp_threads8` / `lgb_mean` の OOF prediction を基準にする。
- exp072 deterministic full replay feature cache を `id` join し、well 内順序を保った window を作る。
- sequence model は validation fold の well を学習に使わない。
- target は `target_tvt - exp073_pred_tvt` の correction。
- 出力は GRU / TCN の fold-out OOF prediction、単体 RMSE、exp073 との誤差相関、alpha blend / ridge blend、distance bucket 別 RMSE。
- 推論 notebook は no-op。`submission.csv` は生成しない。

## 所見

実装のみ完了。Kaggle train は未実行。

参照した既存資料:

- discussion 699289: tabular-only では sequence / spatial context が不足し、PF / Beam や spatial consensus が重要という指摘。
- discussion 699853: CNN/MTP は multi-mode trajectory の発想として有用だが、learned GR matcher は一般化が難しいというコメントがある。
- discussion 707613: PF を NN 化する前に candidate coverage と truth trajectory 近傍生成率を測るべきという指摘。
- discussion 703344: transformer / bfloat16 の copy-task failure。exp088 は float32 固定、AMP 無効。
- public notebook inventory: seq-CNN/CNN 系は低優先度で、現行上位は PF/Beam/TabICL/physical stack が中心。
