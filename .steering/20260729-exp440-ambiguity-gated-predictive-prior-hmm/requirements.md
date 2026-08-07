# 要件

## 依頼

GRから複数のTVTが同程度に支持される曖昧な行では、現在行のGR emissionで
別のTVTへ強制的に更新せず、前行posteriorを物理transitionで進めた
predictive priorを維持するHMM介入を設計する。

ユーザーはGMM自体を要件としておらず、分布族は変更しない。今回は
`KAGGLE_DIRECTION.md`のbacklog、steering、実験ディレクトリを作成して
科学設計を確定する。実装、正規Notebookの作り替え、Kaggle package、
push、run、inference、submissionは行わない。

### 2026-07-29 追加依頼

ユーザーの「exp440を実装してください」により、上記のうちcompact
self-contained実装候補と専用testの作成だけを承認済みに変更する。
正規Notebook採用、Kaggle package、push、Stage 0 run、Stage 1、
inference、submissionは引き続き未承認とする。

### 2026-07-29 実行依頼

ユーザーの「実行してください」により、compact train候補の正規train
Notebook採用、Kaggle package、private CPUでのStage 0 fixed32 push/runを
承認済みに変更する。実行量はscientific candidate 1本、32 HMM well-runs、
保存済み親control再実行0、model/booster/PF/Beam/GPU各0のまま変更しない。
Stage 1、inference、submissionは引き続き未承認とする。

### 2026-07-30 full wells確認依頼

ユーザーの「念のためfull wellsに進んでください」により、Stage 0 FAIL closedを
科学的に撤回せず、変更なしcandidateのfull 773-well OOF確認だけを明示承認する。
Stage 0の実測投影35,365.85秒が既存9時間hard guardを超えるため、773 wellsを
suffix row数のdeterministic LPTで4 CPU shardsへ一意に分割し、各wellを1回だけ
decodeした後にstrict mergeする。scientific candidate 1本、candidate HMM
well-runs 773、保存済み親control再実行0、LightGBM config / trained fold /
booster / fitted model / PF / Beam / GPUはすべて0とする。inferenceとsubmissionは
引き続き未承認とする。

### 2026-07-30 full wells実行結果

4 CPU shardsとstrict mergeを完了した。773 wells / 3,783,989 rows、
candidate HMM 773 well-runs、保存control再実行0でtechnical gateは全PASSした。
candidate RMSE `12.992063`はparent exp209 `11.938287`より`1.053776 ft`悪化し、
positive fold `1/5`、ambiguous-row SSE reduction `-21.3117%`、全6 safety
scopes、by-well p95 / worstもFAILした。`stage1_full_oof_failed_closed`として
rerun、inference、submission、same-OOF rescueなしでterminal closeする。

## 仮説

GRが複数のTVT modeを同程度に支持する行では、current emissionによる更新より、
親transitionで進めたpredictive priorを維持する方がwrong-basin移行を減らす。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 親は`exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`とし、
  TVT/rate state、transition、prior、GR前処理、Gaussian emission、sigma、
  grid、forward-backward、posterior-mean readoutを固定する。
- 曖昧度判定はGMMを使わず、現在行の通常emissionを一度適用した
  provisional filtered TVT marginalの二峰性だけで決める。
- 二峰判定値は`exp236_exact_hmm_posterior_bimodality_audit`の固定値を
  変更せず再利用し、exp440内でthreshold、soft weight、連続gateを探索しない。
- raw GR observed行だけをgate対象とし、GR欠損行は親exp209の処理を維持する。
- ambiguous行ではemission log-likelihoodを`lambda=0`としてneutralizeし、
  filtered分布をpredictive priorへ戻す。`TVT_t=TVT_{t-1}`という点推定の
  hard freezeは行わず、親の物理transitionは維持する。
- gate scheduleはcandidateのcausal forward passだけから作り、
  truth、fold、hidden-like role、persistent episodeを読む前に
  predictionとともにSHA freezeする。
- Stage 0は固定32 wells、scientific candidate 1本、32 HMM well-runs、
  保存済みexp209 control再実行0、model/booster/PF/Beam/GPU各0に固定する。
- Stage 1はStage 0の全technical/mechanism gate PASSと別のユーザー承認を
  必須とし、最大773 candidate HMM well-runs、親control再実行0とする。
- Stage 0/1がFAILした場合、bimodality threshold、lambda、GR scale、
  transition、prior、grid、blend、selector、well/row gateを同じOOFで救済しない。

## 受け入れ基準

- `.steering/20260729-exp440-ambiguity-gated-predictive-prior-hmm/`の
  requirements/design/tasklistがimplementation-only契約を一意に記述している。
- `experiments/exp440_ambiguity_gated_predictive_prior_hmm/`が存在し、
  `config.yaml`、README、SESSION_NOTES、result、metricsが
  `pf_beam`、Stage 0実行状態、inference無効を一致して記録している。
- `KAGGLE_DIRECTION.md`にexp440のStage 0結果とFAIL-closed判断を記録し、
  完了済み項目をアイデアバックログから削除している。
- Stage 0/1のvariant数、HMM well-run数、親control再実行数、
  model/booster/PF/Beam/GPU数、gate、fail actionが事前固定されている。
- compact self-contained train/inference候補と専用testが存在し、
  exp236 detector parity、causal observed-only gate、predictive hold、
  backward schedule固定、truth-late、no-ambiguity parent parity、
  Stage 0/1実行量契約を検証している。
- compact train候補を正規train Notebookへ採用し、Kaggle private CPU
  Stage 0を実行する。inference Notebook、Stage 1、submissionは変更・実行しない。
- deterministic anchorとは扱わず、将来実行する場合にinput、
  ambiguity schedule、prediction、metrics、kernel versionのlogical/content SHAを
  記録する契約がある。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
