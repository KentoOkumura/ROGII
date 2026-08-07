# exp356 結果

## 状態

設計確定、未実装、未実行。scale schedule、CV、LBは存在しない。

## 仮説

exp226 donor covarianceがexp209 constant `sig_r`よりtransition uncertaintyを
校正できるかを0-HMM readoutで判定する。

## 判定予約

Stage 0はNLL 1%以上、4/5 folds、coverage/stress非悪化、fallback/clip各50%以下。
Stage 1はexp209比0.05 ft以上とtail guardを要求する。

## 解釈

現時点では独立親、式、gate、実行量だけが固定されている。

## 次

別承認時だけStage 0を実装する。
