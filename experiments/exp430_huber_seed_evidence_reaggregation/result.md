# exp430_huber_seed_evidence_reaggregation 結果

## 状態

Kaggle CPU preflight v2は12 / 12 technical checks PASS。4 wellのPF、
共通trajectory、1/4 worker readout parity、parent parity、truth-lateを確認した。
preflightはpromotion evidenceではない。承認済み3+1計画のfull 4 shardは
すべてtruth-unread summaryまで完了し、実行量と4 summary SHAを監査・固定した。
truth-late merge version 1はtechnical gateをPASSしたがscientific gateをFAIL。
`huber_seed_evidence_reaggregation_rejected_close_without_rescue`として閉鎖した。

## 仮説

固定128 seedの共通PF軌跡に対してHuber evidenceを使うと、Gaussian evidence
集約よりtailを抑えつつoverall RMSEを改善できる。

## 固定設定

- 親: exp404 / 比較根拠: exp417
- PF: x1.0、500 particles、128 seeds、trajectory variant 1
- candidate: Huber `delta=1.345`、追加clipなし
- aggregation: centered softmax、temperature `5.0`
- validation: exp226の5-fold reporting、fixed scopes、paired by-well tail
- full cost: 773 PF well-runs、98,944 seed-well trajectories、
  49,472,000 particle starts、4 CPU shards
- parent independent full rerun / model / booster / GPU: `0 / 0 / 0 / 0`

## 実装結果

- exp404 PF kernelと同じRNG消費順・軌跡を生成するself-contained trainを実装した。
- float64 trajectory bankをreadout前に凍結し、同じbankからGaussian、Huber、
  arithmetic mean、parent marginal parity readoutを生成する。
- fixed4 preflight、full4 shard、truth-late merge、promotion gateを実装した。
- inferenceは別承認までfail closedとした。
- 専用test `12 passed`、構文、Ruff、Jupytext round-tripを通過した。

## preflight v1

- Kaggle kernel version / id_no: `1 / 128974735`
- runtime: `174.301117 s`
- 4 PF well-runs、512 seed-well trajectories、256,000 particle startsを完了
- trajectory bank raw / logical SHA:
  `b095585a...3985 / 73829c92...4e2`
- 1 worker / 4 workersのprediction/evidence logical SHA: 完全一致
- parent/arithmetic parityの実行中最大差: `0.000484375 ft`
- 保存後のexp430 v1 CSVとexp404 CSVの18,055行最大差: 両列とも`0.0 ft`
- 原因: binary float32とCSV再読込float64 decimalの比較
- 修正: 両辺をexp404保存dtypeのfloat32へ正規化。toleranceと科学設定は不変

## preflight v2

- Kaggle kernel version / id_no: `2 / 128974735`
- runtime: `311.159154 s`
- technical checks: `12 / 12 PASS`
- parent marginal / arithmetic parity: `0.0 / 0.0 ft`
- trajectory / prediction / evidence logical SHA:
  `73829c92...4e2 / 4860ad5a...e39 / ed50a813...f8e`
- summary raw SHA: `3a34add3...ddc`
- v1/v2のtrajectory、prediction、evidence raw SHAは完全一致

## スコア

| メトリック | 値 |
| --- | --- |
| Huber RMSE | 12.992939553 |
| matched Gaussian RMSE | 12.999103257 |
| Huber gain vs matched Gaussian | +0.006163704 ft |
| 保存exp404 temperature-5 RMSE | 10.914522073 |
| Huber gain vs 保存exp404 | -2.078417480 ft |
| arithmetic mean RMSE | 11.594897884 |
| Huber gain vs arithmetic mean | -1.398041670 ft |
| Public LB | 未提出 |
| Private LB | 未提出 |

## full 4 shard

- Kaggle kernel version: shard 0--3すべて`1`
- rows / wells: `3,783,989 / 773`
- PF well-runs / seed-well trajectories / particle starts:
  `773 / 98,944 / 49,472,000`
- truth / fold / hidden-like role / errorのfreeze前アクセス: 全shard `0`
- summary SHA:
  `aac60fd...a74 / d8176e...4d9 / 60a5ea...ff0 / 66d78b...74c`
- scientific contract / preflight summary SHA: 4 shardで一致
- 判定: technical aggregation PASS、scientific gate FAIL

## merge version 1

- Kaggle kernel / version / id_no:
  `kentookumura/exp430-huber-seed-evidence-reaggregation-merge / 1 / 129051025`
- runtime: `397.418685 s`
- technical gate: 11 / 11 checks PASS
- truth / fold / hidden-like role / errorのfreeze前アクセス: 全て`0`
- matched Gaussian比nonworse fold: `4 / 5`
- matched Gaussian比fixed scopes: shallow、raw GR missing、high missingness、
  roughness low、hidden-like 2面が悪化し、all-nonworse FAIL
- paired-well squared-error delta p95: `+0.464221656`
- worst paired-well RMSE delta: `+2.658674657 ft`（well `c3957531`）
- summary / promotion gate / artifact manifest SHA:
  `1e2bbc0b...2870 / ce2993d2...25ad / ae07934c...460e`
- prediction / evidence / global trajectory manifest logical SHA:
  `39cb4f03...213 / 10199e3b...aee / ff671286...bde`

## 再現性

- deterministic anchor: fixed4 preflight v1/v2でtrue
- stable seed: immutable well ID × seed index
- trajectory/evidence/prediction SHA: preflight v1で生成・上記に記録
- kernel version: `2`（technical preflight PASS）

## 解釈

Huber化はmatched trajectory-residual Gaussianの外れ値感度をわずかに改善したが、
平均改善量は小さく、scope/tail悪化を解消しなかった。より重要なのはHuberとmatched
Gaussianの両方が保存exp404 parent marginal evidenceより大幅に悪い点である。
主因はHuber deltaではなく、trajectory residualをseed evidenceにする目的関数と
parent marginal likelihoodの不一致である可能性が高い。

## 次

delta / temperature / clip / scale / particle / seed / filtering尤度、
well/row gateをsame-OOFで救済せず、inferenceとsubmissionを無効のまま閉じる。
保存済みweight ESS・best-seed disagreement・parent marginal weightとの差を使う
0-PF原因分解だけを低優先候補とし、exp430をpositive evidenceには使わない。
