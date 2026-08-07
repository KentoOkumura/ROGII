# exp411_predictive_filtered_rate_innovation_destick

## 状態

- Route: `pf_beam`
- 状態: Stage 0 Version 5完了・mechanism gate FAIL・branch close
- 優先度: P2
- CV / Public LB / Private LB: 未実行
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 原因証拠: `exp408_hmm_message_rate_basin_audit`

## 仮説

exp209のpersistent offsetは、current GR emissionを反映してもrate posteriorが
true-rate変化へ追従できず、absolute datum差を積分することが主因である。
predictive→filtered rate innovationをCUSUMで累積し、同方向の弱い更新が1 rate cell分に
達した区間だけstay massをその方向へ移せば、通常のsticky区間を保ったまま追従遅れを減らせる。

## 親と単一変更

position / rate grid、`sig_r`、`sig_p`、`mom`、GR emission、prior、backward smoothing、
posterior meanはexp209と同じ。変更はtrigger後32 transitionsだけ、stay massの10%を
innovation方向の隣接rate stateへ移す処理だけである。

## 検証方針

- Stage 0: persistent 16 + matched nonpersistent 16のfixed32 mechanism preflight。
- Stage 0実行量: 1 treatment / 32 HMM well-runs、parent HMM rerun 0。
- Stage 1: Stage 0全gate PASSと別承認後だけ1 treatment / 773 HMM well-runs。
- control: 保存済みexp209 prediction / metrics。
- truthはtrigger scheduleとpredictionのfreeze後だけlate joinする。
- Stage 0はlead time、rate方向一致、false trigger、runtimeをAND評価し、小標本RMSEを
  promotion gateにしない。
- Stage 1はexp209比`>=0.05 ft`、4/5 folds、persistent SSE、1000+、hidden-like、
  GR missing、well-tail、fixed blendを全AND評価する。

## 禁止事項

全体`sig_r`拡大、mode ID固定、GR / position / momentum変更、beta / multiplicity変更、
parameter grid、Viterbi / MAP置換、blend、inference、submissionは禁止。

## 実装状態

- fixed32 manifestを実装時に固定した。
  - persistent 16 / matched control 16、5 folds。
  - manifest SHA256:
    `fbbc62b7cb79e16a7fb436f3a9d11f8975e935ad2475a17e2dec4fd7b142e4d6`
- compact self-contained train / fail-closed inference候補をJupytext percent形式で実装した。
- forwardでtrigger後の次rowから32 transitionsをactiveにし、そのscheduleをbackwardで
  同じrow indexに再利用する。
- truth / episodeはfixed32全予測とscheduleのfreeze後だけ読む。
- 専用contract testは18件、exp408回帰込み26件PASS。
- ユーザーの2026-07-27の実行指示により、compact self-contained候補を正規
  train / inference Notebookへ採用した。
- Stage 0 Kaggle CPU実行のみ承認済み。Stage 1、inference実行、submissionは未承認。
- Version 4は32 / 32 HMMを約1,009.6秒・peak RSS約1.02GBで完走し、truth-late前の
  float readback SHA guardで停止した。
- round-trip修正済みVersion 5は同じcanonical kernelで`COMPLETE`。
  32 / 32 wells、`1,133.133秒`、peak RSS `1.020561 GB`。
- technical gateは13 / 13 PASS、生成物の実ファイルSHAと行数もログに一致した。
- mechanism gateは2 / 6 PASS:
  - future-rate方向一致`0.225397 < 0.60`
  - passing folds`0 / 5 < 4 / 5`
  - pre-onset coverage`1.0`、eligible episodes`25`はPASS
  - control active-row fraction`0.136119 > 0.10`
  - persistent-control active-well差`0.0 < 0.20`
- `promotion_eligible: false`、`stage0_fail_closed`。Stage 1、inference、
  submissionは未実行。

## 所見

triggerは十分早く発火した一方、方向一致が低く、persistent / controlを区別せず
全32 wellsでactiveになった。technical契約は全PASSしたため、target-free innovation
CUSUMを因果的な方向triggerに使う固定仮説のnegative resultと判断する。

## 次

exp411のStage 1へは進まない。exp412は先行条件を満たしたが実装・実行は未承認。
同じschedule gateを使うexp420も現行契約のままrunせず、実装参照として保持する。
