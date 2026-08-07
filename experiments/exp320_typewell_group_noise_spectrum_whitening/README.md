# exp320_typewell_group_noise_spectrum_whitening

## 状態

- ルート: `pf_beam`
- 状態: 設計確定・exp311/313待ち・未実装
- 親: `exp311_typewell_group_prefix_suffix_gr_calibration_readout`
- CV / LB: 未実行 / 未提出

## 仮説と変更点

同一Type Well群ではGR residualの自己相関も共通し、independent likelihoodよりshrunk AR(1) innovation likelihoodの方が候補順位を改善する。GR値の平滑化は行わない。

## 検証方針

group AR(1)をglobal/unpooled/group-shuffleと比較し、MRR/top3、4/5 folds、hidden-like、worstを全PASSするまで利用不可。decodeは範囲外。

## 所見

signalを平滑化せず、correlated-noise likelihoodだけの価値を切り分ける。

## 実行入口

notebookは未実装placeholder。`implementation.enabled=false`でKaggle package/push/runは禁止。
