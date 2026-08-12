# 要件

> **閉鎖済み（2026-07-22）**: 親exp323のterminal closeにより未実装・未実行で閉鎖した。新相当実験はexp338と新exp323相当の二段階PASS後に新番号でのみ作成する。

## 依頼

残差rate座標でmomentumを時間変化させる設計を確定する。実装しない。

## 制約

- Route: `pf_beam`。exp323 promotion必須。
- 変更はmomentum scheduleだけ。`mu_r,t`、全sigma、GR、gridを固定する。
- scheduleはparent prior変化とMDだけから決定し、posterior/誤差feedbackを禁止する。
- fixed formula 1本、gridなし、Stage 0 FAILで閉じる。

## 受け入れ基準

- residual coordinate、momentum式、activation、prefix backtest、Stage 1 gateが一意。
- 最大実行量と承認境界が記録される。
- 実装/Kaggle/inference/submissionが無効。
