# exp105_compact_rank_slot_features_on_exp098 結果

## 要約

Kaggle train v1 は完了したが、compact 22-column rank-slot feature set は exp098 full rank-slot より悪化した。提出候補にはしない。

## 仮説

exp098 の rank-slot features から重複・低 utility 列を削ることで、rank-slot idea の有効信号を保ったまま過学習候補を減らせる可能性がある。

## 実行

- Kernel: `kentookumura/exp105-compact-rank-slot-features-on-exp098-train`
- Version: v1
- Output: `experiments/exp105_compact_rank_slot_features_on_exp098/kaggle/output/train_v1`
- Rows / wells: 3,783,989 / 773
- Features: 218 = base 196 + compact rank-slot 22
- Runtime: 11,443.651 sec

## OOF

| model | RMSE TVT |
| --- | ---: |
| `lgb2` | 9.441103161 |
| `lgb1` | 9.477699412 |
| `lgb_mean` | 9.506397523 |
| `lgb0` | 9.774440354 |

best `lgb2` は exp098 `lgb1` 9.358151052 より +0.082952 悪く、exp098 `lgb_mean` 9.427447987 より +0.013655 悪い。exp092 `lgb1` 9.322479896 との差は +0.118623。

## 特徴量

compact rank-slot features は 22 列。上位 importance には `rank1_u_curvature`、`rank2_u_curvature`、`rank3_u_curvature`、`rank1_u_slope`、`rank2_u_slope`、`rank3_u_slope` が残った。一方で、削った列の中にも exp098 の改善に効いていた信号があった可能性が高い。

## Rank Slot 分布

Rank1 は `pf_ancc` 33.65%、`beam_mean` 24.55%、`likpf_mean` 41.80%。`sc_ens` / `hyb` は rank1 / rank2 では 0、rank3 でもほぼ 0 だった。この点は exp098 と同じで、source flag 削減の動機自体は妥当だった。

## SHA

- input cache SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- model manifest SHA: `ae6370fbbe8d1bc37ffdfb7f3c9ad6683bc2b38e3e7f1af72b7c57c840c1a2ca`
- predictions gzip SHA: `397e24aeb15d550ad6fb5c58f8eb4bd462d3c4047d16042ed6cc3b882f9daa0c`
- predictions decompressed SHA: `610154188ad092b8fed9dc60699aa797bd7397e92448d7b5068b6b273fcb374d`
- `lgb2` prediction SHA: `fbe28f97011ce933aa619e500292d4a403d9f457e5100ab77765b8f9028dbe2f`

## 判定

compact 化は rejected。exp098 の全 rank-slot feature set を rank-slot 比較基準として維持する。次に rank-slot を使うなら、exp105 の 22 列単独ではなく、exp092 への add-only merge や top-n candidate-only の別 ablation で検証する。
