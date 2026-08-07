# exp431_affine_ar1_seed_evidence_reaggregation

## 状態

- ルート: `pf_beam`
- 状態: `closed_prerequisite_failed`、未実装、未実行
- CV / LB / Submit ID: なし
- 作成日: 2026-07-28
- PF 親: exp404
- 必須先行実験: exp427

## 仮説

exp427 の affine posterior + fold-safe AR(1) block likelihood が target-free shift識別で完全 PASS するなら、同じ observation family は固定 128 seed PF 軌跡の evidence 集約でも、identity-iid と exp404 Gaussian T=5 を上回れる。

## 固定した設計

- x1.0、500 particles、128 seed の PF trajectory variant は一つ。
- `identity_iid`、`affine_iid`、`identity_ar1`、`affine_ar1` は同一軌跡を採点する。
- candidate は `affine_ar1` のみ。
- 512-row non-overlap block、raw-finite support、prefix affine、outer-fold AR(1) は exp427 をそのまま継承する。
- seed evidence は proper block log-density の総和。mean log-density は監査用のみ。
- centered softmax、temperature 5.0。

詳細は [steering design](../../.steering/20260728-exp431-affine-ar1-seed-evidence-reaggregation/design.md) を正とする。

## 先行条件の結果

exp427 version 2は完走したが、technical / scientific AND gateがともにFAILした。
technicalではeligible block率が`0.721074 < 0.75`、scientificでは
`affine_ar1` MRR `0.386090`がmatched `0.388003`とsaved exp280 `0.388146`
を下回った。このため必須先行条件は成立せず、事前登録どおり本実験を閉じた。

## 実行量

full は 1 PF variant、773 well-runs、98,944 seed-well trajectories、49,472,000 particle starts、4 CPU shards。四 likelihood readout は PF の追加実行ではない。LightGBM/HMM/Beam/GPU は 0、親 control の独立 full rerun は 0。

## 検証方針

条件成立時に予定していたtechnical preflightとfull比較は実施しない。exp427 FAIL後の
support、gate、prior、rho、block、temperatureの変更やsame-OOF rescueも行わない。

## 実行入口

train/inference notebook はterminal closeを示すmarkdown-only placeholderである。
実行コード、Kaggle package、kernelは作成しない。

## 結果

exp427 prerequisite failureにより、未実装・未実行のままterminal closeした。

## 生成物

設計文書と閉鎖記録だけである。trajectory bank、evidence、weight、predictionは
生成していない。

## 所見

exp427のtarget-free readoutではaffine + AR(1)がmatched / saved controlを
上回らず、coverage gateもFAILした。PF seed選別へ転用する根拠は成立しなかった。

## 次

なし。exp431を再開しない。
