# 要件

> **閉鎖済み（2026-07-22）**: 親exp323のterminal closeにより未実装・未実行で閉鎖した。新相当実験はexp338と新exp323相当の二段階PASS後に新番号でのみ作成する。

## 依頼

現行floorを踏まえた時間変化`sig_p,t`の低優先実験を設計確定する。実装しない。

## 制約

- Route: `pf_beam`。exp323 promotionと上位案の判断後にのみ候補化する。
- 0.1225未満のsigma、grid step変更、rate/GR/momentum変更は禁止。
- formulaはgrid quantizationだけから決定し、posterior/誤差feedbackを使わない。
- Stage 0 FAILで閉じ、sigma gridを行わない。

## 受け入れ基準

- floor、上限、式、prefix NLL、activation/clip gateが固定される。
- Stage 0/1実行量と承認境界が明記される。
- 実装/Kaggle/inference/submissionが無効である。
