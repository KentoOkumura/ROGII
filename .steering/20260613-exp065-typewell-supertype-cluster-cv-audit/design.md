# 設計

## アプローチ

1. `data/raw/train/*__typewell.csv` を読み、byte hash、行数、TVT/GR 範囲、GR 欠損率をまとめる。
2. 各 typewell の GR を TVT 正規化軸に resample し、median fill、z-score、軽い smoothing をかけた固定長 signature を作る。
3. exact hash duplicate group を baseline として保存する。
4. native `typewell.csv` の GR 列を quantize し、rolling k-gram hash で同一部分列候補を作る。候補 pair/lag について overlap rows、exact match rate、row lag、ft 換算、containment 関係を保存する。
5. shifted NCC は TVT signature を左右にずらし、overlap が十分ある範囲で最大相関と shift を記録する。全ペアから各 well の上位 `top_k` と閾値以上のペアを残す。
6. constrained DTW は shifted NCC の候補ペアだけに実行する。Sakoe-Chiba band で計算量を抑え、z-score GR signature 間の平均 absolute cost を similarity に変換する。
7. exact / native_overlap / shifted_ncc / dtw それぞれで閾値別の connected components を作り、common typewell candidate group として保存する。

## 実験範囲

- 対象実験: `exp065_typewell_supertype_cluster_cv_audit`
- Route: `pf_beam`
- 親実験: `studies/typewell_group_audit.py`
- 変更する変数: typewell 共通性の定義。exact CSV hash から shifted NCC / DTW 類似グループへ拡張する。
- 固定する変数: モデル、OOF、postprocess、提出 flow は扱わない。

## リスク

- リークリスク: train/test 予測や target を使わず typewell CSV のみを読む。後続で特徴量化する場合は fold-safe neighbor pool が必須。
- CV/LB 不一致リスク: この実験は CV/LB を出さない。共通 typewell 候補の発見だけを成果物にする。
- ランタイム/メモリリスク: DTW は全ペアではなく shifted NCC で絞った候補に限定する。signature 長、band、top_k は config 化する。
