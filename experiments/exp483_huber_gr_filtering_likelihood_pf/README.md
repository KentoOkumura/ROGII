# exp483_huber_gr_filtering_likelihood_pf

## 状態

- Route: `pf_beam`
- 状態: Stage 1 technical PASS / scientific FAIL、terminal close
- CV: `11.095404595`
- LB / Submit: なし
- 親: exp417、実装参照・保存control: exp404

## 仮説

exp389のfixed Huber `delta=1.345`を粒子filtering尤度へ直接入れると、
外れGRが正しい軌道modeを消す現象を抑えられる。

## 変更点

exp404 x1.0 PFのGaussian particle log emissionだけをHuberへ置換する。
exp430の凍結軌道seed evidence再集約とは異なり、weight、ESS、resampling、
以後のparticle trajectoryが変わる。

## 検証方針

Stage 0はstable-hash fixed32のtechnical preflight。全PASSと別承認後だけ、
1 variant ×773 wells ×128 seeds ×500 particlesを評価する。保存exp404
scale-5 x1.0をcontrolとし、control PFは再実行しない。

## 実装済み範囲

- Jupytext percent形式のcompact self-contained train候補
- fixed Huber `delta=1.345` particle filtering kernel
- exp404 Gaussian reference / no-op parity / stable seed contract
- fixed32 target-free prediction、ESS/resampling ledger、SHA freeze/readback
- freeze後だけのtruth / 保存control report-only readout
- fail-closed technical gateと専用test

compact候補を正規train Notebookへ採用し、Stage 0と全773 wellsのStage 1
truth-late CVを完了した。inference、submissionは無効。

## Stage 0結果

- technical gate: `10 / 10 PASS`
- candidate RMSE（fixed32 report-only）: `9.811671590 ft`
- 保存exp404 control RMSE（同一fixed32）: `9.616740808 ft`
- candidate - control: `+0.194930782 ft`
- improved wells: `18 / 32`
- runtime / full projection / peak RSS:
  `283.781 sec / 6,855.083 sec / 0.495358 GB`

fixed32値はCVでもpromotion判定でもないが、参考診断では悪化した。

## Stage 1結果

- technical gate: 全PASS
- candidate / 保存exp404 control:
  `11.095404595 / 10.914522073 ft`
- candidate - control: `+0.180882522 ft`悪化
- 改善fold: `3 / 5`
- improved / worsened wells: `369 / 404`
- by-well p95 / worst regression:
  `+0.520909635 / +33.458522531 ft`
- fixed exp209 HMM/PF 50:50: `+0.077245509 ft`悪化
- runtime / peak RSS: `12,454.354 sec / 3.566319 GB`
- decision: `terminal_close_without_huber_or_pf_rescue`

## 所見

### 良い点

exp389では平均`0.085546 ft`、5/5 foldsのpositive signalがある。

### リスク

exp389はtail gateをFAILし、exp430もpost-hoc Huber evidenceで改善しなかった。
Stage 1でもHuber filtering likelihoodは3 foldsと一部scopeを改善したが、
fold 4、GR observed、長いsuffix、well-tailを悪化させた。wrong basinに入る
wellの大幅悪化を防げず、Gaussian PFを置き換える根拠にはならない。

## 成果物

compact source / 正規train Notebook / 専用testを実装し、Kaggle version 1で
Stage 0、version 2でStage 1 prediction、metrics、gate、audit、runtime生成物を
作成した。

## 次

事前登録どおりHuber/PF/blend/selectorのsame-OOF救済を行わずbranchを閉じる。
inferenceとsubmissionは実行しない。
