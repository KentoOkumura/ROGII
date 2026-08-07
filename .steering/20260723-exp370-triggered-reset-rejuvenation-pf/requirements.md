# 要件

## 依頼

PF 状態変数案として triggered reset rejuvenation PF の backlog、実験ディレクトリ、
steeringを作り、実装前の設計を確定する。

## 制約

- Route: `pf_beam`
- `docs/06_reproducibility.md` のPF seed / SHA規約に従う。
- 実装、Notebook置換、Kaggle実行、推論、提出は行わない。
- exp231負結果を解除せず、独立trigger/coverage Stage 0を必須にする。
- atlasはproposalだけ。直接予測・emission・oracleには使わない。

## 受け入れ基準

- trigger、atlas、topK、注入率、配分、jitter、refractoryが一意である。
- Stage 0/1の実行量、truth freeze、全gate、fail policyが固定されている。
- stable RNG、fold-safe atlas、train/test別生成を記録する。
- 既存未採番backlogをexp370へ置換し、summaryへ登録する。

## 2026-07-25 実装承認

- ユーザーの `exp370を実装してください` を Stage 0 実装の承認として記録する。
- 実装対象は500 particles × 1 seed × 773 wellsのStage 0 diagnosticだけに限定する。
- Stage 1の10%粒子再注入PFは、Stage 0の全gate PASSと別承認が必要。
- Kaggle package / push / run、正規Notebook採用、推論、提出は引き続き未承認。
- exp231のatlas実装を内部表現の参照元とし、256行窓を32点へ圧縮、source stride 32、
  2 ft TVT bin、well/bin最大6 patch、bin当たり2 outer-train wells以上を固定する。
- bad-event horizonはsource-ageと同じ128行、circular control shiftはrefractoryと同じ
  512行とする。
- bad-event labelとbase coverageは保存済みexp072 `likpf_mean`を使い、Stage 0の
  1-seed PF predictionはESS診断専用とする。

## 2026-07-25 Stage 0実行承認

- ユーザーの `実行してください` を、compact train候補の正規Notebook採用と
  private Kaggle CPU Stage 0 package / push / runの承認として記録する。
- 実行量はdiagnostic PF replay 1、500 particles、1 seed、773 wells、
  773 seed-well runs、5 reporting foldsに固定する。
- scientific variant、full parent PF control replay、LightGBM config、trained fold、
  boosterはいずれも0とする。
- Stage 1、inference、submissionは承認対象外で、引き続きfail-closedとする。
