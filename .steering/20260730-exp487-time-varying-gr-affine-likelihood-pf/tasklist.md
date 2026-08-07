# タスクリスト

## 未着手

- current-test deployment process-noise tableはinference承認時に生成・検証する。
- Stage 1はStage 0全PASS後も別承認を得る。

## 進行中

- なし。

## ブロック中

- Stage 1、inference、submissionは未承認。

## 完了

- exp345/350/211/404/417の差分と結果を確認した。
- causal/RTS scheduleのPF emission適用、実行量、gate、再現性を確定した。
- backlog、steering、design-only scaffoldを作成した。
- exp345 causal schedule parityとexp350 RTSをcompact self-contained train候補へ実装した。
- prefix fit、outer-fold process noise、EKF/RTS、missing/fallback、Joseph covariance、
  affine emission、truth-late、SHA freeze、Stage 0/1 gateを実装した。
- exp404 identity-affine bitwise parityを含む専用testを追加した。
- fail-closed compact inference候補を追加した。
- 2 variants、64 PF well-runs、8,192 seed-well、4,096,000 particle starts、
  control/HMM/Beam/model/booster/GPU rerun 0を再確認し、Stage 0承認を記録した。
- compact train候補をcanonicalへ採用し、strict Kaggle packageを生成した。
- canonical Kaggle CPU kernel version 5（id_no `129180524`）でStage 0を完了した。
- fixed32 156,088行・32 wellsでcausal `12.634360`、RTS `13.391424`、
  saved exp404 control `9.616741` RMSEを記録した。
- 64/64 variant-wellsのtruth前freeze、禁止情報のfreeze前読込0、runtime、
  RSS、schedule / RTS / covariance / seed / SHAを含む全15 technical checksがPASSした。
- Stage 0完了後にpush / Stage 0実行フラグをOFFへ戻し、Stage 1は別承認待ちとした。
