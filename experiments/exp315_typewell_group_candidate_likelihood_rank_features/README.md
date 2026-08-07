# exp315_typewell_group_candidate_likelihood_rank_features

## 状態

- ルート: `ml_model`
- 状態: 設計確定・exp312/313待ち・未実装
- 親: `exp264_exp263_candidate_confidence_dual_selector`
- CV / LB: 未実行 / 未提出

## 仮説と変更点

群別GR emissionはcandidate valueを直接補正するより、candidate rank/margin/entropyとしてselectorへ渡す方が安全に利用できる。exp293 deployable12とcorrected exp264を固定し4列だけadd-onlyする。

## 検証方針

Stage Aは0-model rank readout、PASS時だけStage Bで40 nested selector models。MRR/fold gate、親比gain、hidden-like、worst guardを全PASSするまでinference不可。

## 所見

GR likelihoodが候補選択に寄与しない場合はStage Aで止め、selector costを使わない。

## 実行入口

notebookは未実装placeholder。`implementation.enabled=false`で学習/push/runは禁止。
