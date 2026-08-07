# 要件

## 依頼

「exp226本体を直す場合」として、support-aware local-linear shrinkageを設計確定する。実装は行わない。

## 制約

- Routeは`pf_beam`、親は保存済みexp226とする。
- exp329 Stage 0全gate PASSとsupport-risk contract SHA固定を必須依存とする。exp329 Stage 1は依存しない。
- 変更はtarget-well raw/smoothed local-linear interceptだけとする。
- fallbackは同じk50 donor・同じkernel weightのweighted constantとし、zero drift/Z-onlyは禁止する。
- risk q80以上だけ連続発火し、最大shrinkは50%の1式に固定する。
- 保存済みfold kappaを使い、kappa再fit、K/bandwidth/ridge/bucket/ANCC/GR/Uの変更を禁止する。
- HMM varianceを変えるexp324、K-scale selector、error/bias transferと分離する。
- full OOF前に32-well parent parityを必須とし、各段階を別承認にする。

## 受け入れ基準

- parent/fallback/risk/activation/shrink式と不変条件が一意に定義される。
- parent kappa file SHA、parity tolerance、実行量、promotion/停止条件が記録される。
- configでimplementation/Kaggle/inference/submissionがすべて無効である。
