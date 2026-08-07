# exp111_learned_pf_observation_likelihood_probe

## 状態

Kaggle train v1 完了。提出なし。

## 仮説

exp099 の hand-crafted multi-observation likelihood は oracle headroom を増やしたが、top1 scorer としては崩壊した。exp101 の supervised row-wise selector も `likpf_mean` 単体を超えなかった。

そのため候補 index を直接選ばず、PF/Beam/likPF 候補ごとに「真値から 10ft 以内に入る確率」と期待誤差を learned observation likelihood として校正する。改善した場合も提出候補ではなく、PF weight ablation や ML add-only feature の材料にする。

## 検証方針

- 入力: exp099 v2 train feature cache
- 候補: `pf_ancc`, `beam_mean`, `likpf_mean`, `sc_ens`, `hyb`
- 形式: candidate-long
- 学習器: LightGBM binary within10 classifier / L1 expected error regressor
- split: GroupKFold by `well`、`run_folds=1` smoke
- 主指標: candidate likelihood AUC、logloss、brier、calibration、topK coverage
- 補助指標: learned top1 diagnostic RMSE、bucket metrics

## 所見

learned likelihood は candidate-level within10 AUC 0.913327 で、exp099 hand-crafted `multiobs_score` AUC 0.617355 を大きく上回った。topK coverage も改善したため、PF weight / ML feature follow-up は支持する。

ただし diagnostic top1 は within10 が `likpf_mean` 単体より悪く、direct replacement にはしない。

## 注意

この実験は train-side likelihood audit 専用で、`submission.csv` は作らない。learned likelihood の top1 をそのまま hidden test prediction に使わない。
