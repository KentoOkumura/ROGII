# exp115_hidden_like_spatial_holdout_from_ppt 結果

## 仮説

公式 PPT slide10 の赤い Verification well 分布を空間 proxy として読み取り、train wells から隠しテストに近い固定 holdout を作る。これは exact hidden split 復元ではなく、`exp092` / `exp073` / `exp098` などの ML route anchor を追加監査するための評価面である。

## 設定

- 親: `exp092_u_projection_correction_disagreement_fullrun`
- 診断親: `exp044_stratified_groupkfold_cv_audit`, `exp065_typewell_supertype_cluster_cv_audit`, `exp073_gpu_reproducibility_guard_for_exp063_full_replay`, `exp098_selector_rank_slot_features_on_exp073`
- 検証: `hidden_like_spatial_holdout_from_official_ppt_slide10`
- メトリック: 今回はスコアなし。holdout well count、PPT red distance、分布レポートを保存する。
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| PPT red component | 45 |
| PPT red pixels | 6,217 |
| Train wells | 773 |
| `verification_like_spatial` valid wells | 200 |
| `verification_like_typewell_purged` valid wells | 200 |
| Purged train excluded wells | 16 |
| Spatial median PPT red distance | 0.018609910 |
| Spatial max PPT red distance | 0.080668542 |
| CV | - |
| Public LB | - |
| Private LB | - |

## 再現性

- deterministic anchor: false。submission anchor ではない。
- seed policy: 乱数なし。well_id、PPT distance、component order の deterministic sort。
- kernel version: `kentookumura/exp115-hidden-like-spatial-holdout-from-ppt-train` v1、status `COMPLETE`。
- PPTX SHA256: `c083c59df01f0fdf1fea860bc977fcdcf278eb12a33282a0200044c8abf950fc`
- Slide image SHA256: `e8ed99d562ae0a630a1780ac4bdb73cf2c5fd3f40d733723ddaece80f5e17901`
- model SHA / manifest SHA: モデルなし。
- prediction SHA: 予測なし。
- submission SHA: 提出なし。
- rerun result: local smoke 2 回と Kaggle train v1 で red component count 45、valid wells 200/200、purged excluded 16 が一致。

## 解釈

PPT slide10 の赤 component は 45 件として抽出でき、公式資料の Verification 分布に寄せるための target points として使える。生成された `verification_like_spatial` は train 773 wells から 200 wells を選び、`verification_like_typewell_purged` は同じ valid count を保ちつつ exact typewell group の train mate 16 wells を purge 対象にした。

この holdout は CV/LB の代替ではなく、現行 anchor や後続 feature 実験が「見えない test well に近い空間分布」で崩れないかを見る stress readout として使う。

## Kaggle output

- 保存先: `experiments/exp115_hidden_like_spatial_holdout_from_ppt/kaggle/output/train_v1/`
- `holdout_wells.csv`: 400 rows + header
- `fold_assignments.csv`: 773 rows + header
- `ppt_red_points.csv`: 45 rows + header

## 次

- follow-up として `hidden_like_anchor_score_readout_on_exp115` を作り、保存済み Kaggle output の `fold_assignments.csv` に `exp092` / `exp073` / `exp098` の OOF prediction を merge して holdout RMSE、bucket、worst-well delta を読む。
