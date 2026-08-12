# 要件

> **閉鎖済み（2026-07-22）**: 親exp323のterminal closeにより未実装・未実行で閉鎖した。新相当実験はexp338と新exp323相当の二段階PASS後に新番号でのみ作成する。

## 依頼

exp226 donor共分散からsegment別`sig_r,t`を作るHMM内部改善を設計確定する。実装しない。

## 制約

- Route: `pf_beam`。exp323全gate PASSが必須。
- 変更はrate diffusion varianceだけ。rate prior平均、GR、momentum、position、gridを固定する。
- outer-fold donorだけを使い、sigma scheduleをtruth-freeに凍結する。
- fixed 1 formula、parameter gridなし、Stage 0 FAILで閉じる。

## 受け入れ基準

- robust covariance、shrink、clip、fallback、calibration gateが一意に定義される。
- Stage 0/1の実行量と承認境界が記録される。
- 実装/Kaggle/inference/submissionが無効である。
