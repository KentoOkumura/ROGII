# exp343 ACF effective-sample likelihood tempering audit

## 状態

- Route: `pf_beam`
- 状態: Stage 0 Kaggle version 1完了、固定gate FAILでbranch close
- 優先度: P3 closed
- 親実験: `exp281_exp226_residual_offset_exact_hmm_transition_probe`
- 履歴参照: `exp320_typewell_group_noise_spectrum_whitening`
- Kernel: `kentookumura/exp343-acf-effective-sample-tempering-train`

## 仮説

既知prefixの連続finite residualからwell別の積分自己相関時間を推定し、
emissionを`1/tau_eff`で弱めれば、相関したGR行を独立観測として過剰累積する
問題を近似的に緩和できるかを調べた。

旧exp320のgroup AR(1) priorは使わず、current well ACFとouter-train fold
medianだけで構成した。exp311/313のType Well群transferとは独立した監査である。

## 検証方針

- lag 1--20、正のACFだけを加算し、欠損境界をまたぐpairは禁止。
- finite residual 128未満または各lag 20 pair未満はfold priorへfallback。
- raw tauをsupport 200でouter-train fold中央値へlog-space shrinkし、`[1,4]`へclip。
- full-prefix対last-512は両windowがraw-evaluableなwellだけで安定性を評価。
- Stage 0全gate通過時だけ、別途承認後にGaussian emissionへwell単位の
  `1/tau_eff`を掛けるStage 1を許可。
- Student-t/Huber、missing補正、state別tau、lag/support/clip gridは禁止。

## 所見

Kaggle private CPU version 1を`273.66704466799996 sec`で完了した。
773 wellsのうちjoint-evaluableは295（38.16%）、fallbackは478（61.84%）。
raw tau fold priorはfullで約9.77--10.04、tailで約24.26--25.17となり、
`tau_eff`はfull 99.74%、tail 100%で上限4へclipされた。Spearmanは定数列のため
未定義、stable foldは0/5だった。

固定gateはFAIL。well別tempering係数の安定推定に失敗したnegative resultとして
branchを閉じた。clipで生じた一律`tau_eff=4`をStage 1へ持ち込まず、救済grid、
HMM、inference、submissionも実施しない。

## 実行量

- Stage 0: deterministic diagnostic 1、reporting folds 5
- HMM well-run / model config / trained fold / booster: `0 / 0 / 0 / 0`
- 親control再実行: 0
- Stage 1 / inference / submission: 未実装・未実施

## 文書

- Steering: `../../docs/legacy/steering/20260722-exp343-acf-effective-sample-likelihood-tempering-audit/`
- 詳細結果: `result.md`
- 設定: `config.yaml`
