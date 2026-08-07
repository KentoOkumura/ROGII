# exp425_symmetric_datum_reanchor_exact_hmm

## 状態

- ルート: `pf_beam`
- 状態: Kaggle Stage 0完了・`stage0_fail_closed`
- 優先度: P3・高リスク機構実験
- CV / Public LB / Private LB: なし
- 作成日: 2026-07-28
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 機構証拠: `exp408_hmm_message_rate_basin_audit`、`exp412_beta_filter_rate_disagreement_two_pass_reset`
- Kaggle kernel version / id_no: `1 / 128930925`

## 仮説

exp209のpersistent offsetは、local rateが再同期してもabsolute datumが戻らない
translation-gauge lockによって維持される。persistent beta-filter rate disagreementを
target-freeな発火条件とし、rate状態を変えずにabsolute datumだけ異なる3枝を生成すれば、
将来GR尤度が正しいdatum枝へsoftに質量を移せる可能性がある。

## 固定した変更

- unchanged exp209 first passで最初のpersistent disagreement eventをfreezeする。
- event transitionで`negative / parent / positive`の明示的datum branchを生成する。
- branch priorは`0.10 / 0.80 / 0.10`。
- shiftは`±max(filtered position std, 0.35 ft)`。
- eventは1 wellにつき最大1回。
- rate gapの符号はdatum方向の選択に使わない。
- rate grid / transition、position noise、GR emission、prior、support、readoutは固定する。
- exact sum-productで3枝をsoft周辺化する。

## Stage 0方針

exp412の固定32 wellsをmechanism sampleとして再利用する。

- baseline first pass: 32 HMM well-runs
- treatment: 32 logical HMM well-runs、3 branch states
- active scientific variant: 1
- LightGBM / model / booster / PF / Beam / GPU: 0

backward-cause SSE、forward / control安全性、soft datum方向、branch posterior mass、
runtime / memoryを事前固定AND gateで判定する。固定32はCVやpromotion evidenceには使わない。

## 検証方針

- reporting fold: 親と同じ5 folds
- group: `well_id`
- truth join: event、shift、branch posterior、predictionをfreezeした後だけ実施
- scientific gate: soft datum方向、backward-cause SSE、forward / control安全性
- technical gate: parent parity、finite、normalization、runtime、memory、SHA
- fail policy: trigger、shift、prior、gate、readoutを同一OOF上で救済せず閉じる

## 所見

Kaggle private CPU Stage 0は2,684.506秒で完了した。technical gateは12 / 13 PASS
だったが、full投影`64,847.602秒 > 30,600秒`でruntime gateをFAILした。

mechanism gateは3 / 7 PASS。soft datum方向一致は`0.396578 < 0.60`、passing foldは
`1 / 5 < 4 / 5`、backward-cause SSE削減は`0.000698 < 0.10`、matched-control
reanchor massは`0.285635 > 0.10`だった。forward safety、control pooled RMSE safety、
active reanchor massはPASSしたが、主目的と選別性を満たさない。

対称3枝とexact full-sequence evidenceでもabsolute datum方向を識別できず、controlにも
branch massを配った。same-sampleでtrigger、prior、shift、readout、gateを救済せず、
`stage0_fail_closed`として閉じる。

## 実装状態

- compact self-contained train / inference Jupytext source: 実装済み
- 正規train / inference Notebook: Jupytext sourceから採用済み
- exact HMM:
  zero-shift parent、first persistent event、対称3 conditional branch、
  exact evidence marginalizationを実装済み
- test:
  exp209 parity、parent-only parity、single-event、soft branch、truth-late、
  inference fail-closeを実装済み
- Kaggle package / push / Stage 0 run: version 1完了
- Stage 1 eligible: false
- Stage 1 / inference / submission: 未実施・禁止

詳細な契約は
`.steering/20260728-exp425-symmetric-datum-reanchor-exact-hmm/`と
`config.yaml`を正とする。

## 次

現行branchは閉鎖する。Stage 1、inference、submissionへ進まない。固定32は
mechanism sampleであり、結果をCVや一般化根拠とは扱わない。
