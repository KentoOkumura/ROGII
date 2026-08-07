# exp153_full_rank_slot_addonly_on_exp092 結果

## 状態

Colab L4 high-memory で full train 完了。`latest_done_summary.json` を確認済み。

## 目的

exp092 の U-projection correction / disagreement surface に、exp098 の full rank-slot feature groups を add-only で追加し、exp092 に対して非重複の改善が残るか確認する。

## 実装

- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- rank-slot source parent: `exp098_selector_rank_slot_features_on_exp073`
- variant: `u_projection_full_rank_slot_addonly`
- base features: exp072/exp073 full replay 196 features
- exp092 U-projection groups: `projection_correction`, `u_disagreement`
- rank-slot groups: `rank_slot_delta`, `rank_slot_identity_score`, `rank_slot_u_projection`, `rank_slot_u_disagreement`
- candidate path replacement: なし
- Colab runner: `exp153_full_rank_slot_addonly_on_exp092_colab_train.ipynb`

## GPU コストガード

- active variant 数: 1 (`u_projection_full_rank_slot_addonly`)
- LightGBM config 数: 3 (`lgb0`, `lgb1`, `lgb2`)
- fold 数: 5
- 合計 booster 数: 15
- exp092 control 再学習: なし
- baseline は保存済み exp092 `lgb1` CV 9.322479896 / Public LB 8.350 を参照する。

## OOF

- run id: `run_20260628_133009_l4_highmem_local_cache`
- rows: 3,783,989
- features: 304
- active variant: `u_projection_full_rank_slot_addonly`
- active mode: `gpu_repro_guard_dp_threads8`
- lgb0 pooled RMSE: 9.630078943598752
- lgb1 pooled RMSE: 9.388317392953782
- lgb2 pooled RMSE: 9.413605687155531
- lgb_mean pooled RMSE: 9.423385453890534

## 判断

exp092 best `lgb1` CV 9.322479896、exp092 `lgb_mean` CV 9.343064066 に対して悪化した。full rank-slot add-only は、このままの形では採用しない。

exp098 rank-slot feature groups を全量 add-only しても、exp092 の U-projection correction / disagreement surface に対する非重複な改善は確認できなかった。特に `lgb_mean` は 9.423385454 で、exp139 small add-only の 9.324907641、exp147 replacement-only の 9.397013393 と比べても弱い。

次に進むなら、full group 全量追加ではなく、feature importance / by-well / bucket を見て、rank-slot 系の局所的に効く列だけを絞る方向にする。
