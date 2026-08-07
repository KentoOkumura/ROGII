# 要件

> **閉鎖済み（2026-07-22）**: 親exp308のterminal closeにより未実装・未実行で閉鎖した。exp338 chainへ接続せず、再検証はexp209直系の`exp345_exp209_time_varying_gr_affine_calibration_hmm`で管理する。

## 依頼

GR affine係数`a_t,b_t`を時間変化させるHMM内部改善を、既存group-prior枝と分離して設計確定する。実装しない。

## 制約

- Route: `pf_beam`。旧設計ではexp308 PASSを前提としていたが、その条件は成立せずterminal close済み。
- current well、frozen parent path、raw GR、Type Wellだけを使い、group priorを禁止する。
- causal one-pass filter、schedule freeze、variant HMM 1回に限定する。
- joint state、smoother、反復、grid、transition変更、blendを禁止する。
- prefix mask、worst、fallback、8.5h runtime gateを必須にする。

## 受け入れ基準

- state、初期fit、process-noise rule、observation variance、fallback、truth boundaryが一意。
- microbenchmark/Stage 0/Stage 1の実行量と個別承認が記録される。
- 実装/Kaggle/inference/submissionが無効。
