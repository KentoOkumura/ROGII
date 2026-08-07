# exp065_typewell_supertype_cluster_cv_audit

## 状態

- ルート: pf_beam
- 状態: implemented
- CV: なし
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-13
- 親実験: `studies/typewell_group_audit.py`

## 仮説

Exact duplicate CSV だけでは説明できない共通 typewell が、GR 曲線の shifted NCC または constrained DTW 類似グループとして見つかる可能性がある。

## 変更点

- exact hash duplicate group を作る。
- resampled GR signature で shifted NCC 類似ペアを抽出する。
- NCC 候補ペアに constrained DTW をかける。
- exact / shifted NCC / DTW の閾値別 connected components を common typewell candidate group として保存する。

## 検証方針

- Fold: なし
- Group: discovered common typewell candidate group
- Stratification: なし
- Leakage Check: train typewell CSV のみを読み、horizontal target / OOF / submission は使わない。

## 実行入口

- 学習 notebook: `exp065_typewell_supertype_cluster_cv_audit_train.ipynb`
- 推論 notebook: `exp065_typewell_supertype_cluster_cv_audit_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp065_typewell_supertype_cluster_cv_audit EXTRA_ARGS="--notebook train --run-on-push --strict"`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | なし |
| Public LB | - |
| Private LB | - |

## 所見

- Kaggle train v1 completed。
- exact hash は 752 unique groups / 34 duplicate wells。
- native exact overlap は 54 groups / 41 multi-well groups / 760 wells / max group 71。
- `028d7b28` と `0dd99dc5` は lag 218 rows = 109 ft、1774 rows が GR 完全一致。
- `shifted_ncc >= 0.98` は 103 multi-well groups / 314 wells。
- `dtw_similarity >= 0.94` は 63 multi-well groups / 167 wells。
- 57 groups 付近は native row-lag overlap で近く再現できる。

## 次

- `typewell_neighbor_prior_features` では、exact hash、native row-lag overlap、high-NCC、high-DTW を別々の fold-safe neighbor pool として比較する。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
