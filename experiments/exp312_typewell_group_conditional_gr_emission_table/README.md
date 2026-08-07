# exp312_typewell_group_conditional_gr_emission_table

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU version 1完了・gate FAIL・branch closed
- 親: `exp311_typewell_group_prefix_suffix_gr_calibration_readout`
- CV / LB: MRR gain `-0.001592`、top3 gain `-0.002444` / 未提出

## 仮説と変更点

群共通性をscalar affineではなく、Type Well GR decile・|gradient| tertile・horizontal欠損flagに条件づけたshrunk Student-t residual tableで表すと、物理候補のGR順位が安定する。decodeや予測は作らない。

## 検証方針

outer-trainだけでtableをfitし、exp293の固定deployable12に対するtruth-nearest MRR/top3、group shuffle、matched TVT shift、fallback率を5 foldsで読む。candidate順位とtableをSHA凍結した後だけouter-valid truthを結合する。

## 所見

global-unconditional baselineに対しconditional realはMRR `0.336112 → 0.334519`、top3 `0.358090 → 0.355646`と平均で悪化し、改善foldは`0/5`だった。real-shuffle差も`+0.001611`で固定閾値`+0.02`に届かず、hidden-like 2面も非悪化を満たさなかった。fallback率`1.823%`とlate-truth境界だけはPASSした。

## 実行入口

compact trainを正規notebookへ採用し、`kentookumura/exp312-typewell-gr-emission-rank-audit-train` version 1（id_no `128090149`）で実行した。private CPU / internet off、scientific 1 + controls 2、5 folds、model/booster/decoder 0、runtime `326.623 sec`。

## 次

固定gate FAILのためbin/df/kを救済せずbranchを閉じる。exp313〜320は停止を維持し、この結果を根拠に追加の同系救済backlogは作らない。
