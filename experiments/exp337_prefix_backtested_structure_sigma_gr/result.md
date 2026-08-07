# exp337_prefix_backtested_structure_sigma_gr 結果

## 状態

Kaggle CPU Stage 0 version 1完了。固定gateをFAILし、Stage 1へ進まず枝を閉じた。CV、LB、prediction、submissionはない。

## 結果

| origin | 評価finite pairs | finite-only NLL | zero-fill NLL | structure-added NLL | structure勝利 vs finite / zero-fill |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.60 | 186,184 | 3.027165 | 3.589239 | 3.073866 | 0/5 / 5/5 folds |
| 0.80 | 178,469 | 2.971854 | 3.571889 | 3.015784 | 0/5 / 5/5 folds |

- 773 wellsすべてを両originで評価し、fallbackは0だった。
- structure-addedのzero-fill比gainはorigin 0.60で`0.515373`、0.80で`0.556105` per finite residualだった。
- full-prefix median `tau_structure`は`0.0`で、事前gate `>=5.0`をFAILした。
- lower clipは`42/773 = 0.054334`で、上限0.10以内だった。

## 解釈

zero-fill scaleはforward residualに対して広すぎ、構造分散を加えた候補はそれより改善した。しかしfinite-onlyが両origin・全foldで優位であり、典型wellでは推定構造分散も0だった。このため、typewell不一致やalignment errorを独立の追加分散として扱う中心仮説は支持されない。

事前固定どおり、同じ結果上のsplit、threshold、scale、likelihood救済を行わず終了する。Stage 1 exact-HMM、inference、submissionは未実装・未実行のままとする。

## 再現性

- Kaggle kernel: `kentookumura/exp337-prefix-backtested-structure-sigma-gr-train` version 1、id_no `128220965`
- runtime: `143.899363 sec`、CPU / internet off
- scientific contract SHA: `57fa5c9e3def170f8a3a83018eb4d69ab69ef835f5b61511633c066189feddb5`
- input dependency contract SHA: `7f19db5ec37b524b9caf17beeee30b281f11760fb6061a1dfe4a3bccc9cbef32`
- rolling-origin audit content SHA: `3f72fb1dcb4ea95c4b77b54d2b75c7f302dd0fd7fbd2707ddcd2c5532dd0e883`
- full-prefix audit content SHA: `b83a5a6c41dc6887afec2840c160482091aafcf26fd19de5f5167487888fd2b8`
- unknown-suffix truthのfreeze前read: false
- output archiveは取得せず、Kaggle logsとoutput file listで確認した。
