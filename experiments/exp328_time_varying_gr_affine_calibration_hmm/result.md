# exp328 結果

## 状態

親exp308のterminal closeにより閉鎖済み。未実装・未実行で、runtime、prefix mask、GR NLL、HMM RMSE結果はない。

## 固定した判定

projected runtime 8.5時間以下、prefix mask RMSE 0.05 ft以上、4/5 folds、GR NLL改善、boundary jump p95 `<=3σ`、hidden-like/worst/fallback guardを全要求する。full suffixも親比0.05 ftとtail hard guardを要求する。

## 次

本実験は再開しない。再検証はexp209直系の独立した`exp345_exp209_time_varying_gr_affine_calibration_hmm`で管理する。
