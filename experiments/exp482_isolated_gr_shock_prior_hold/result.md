# exp482_isolated_gr_shock_prior_hold 結果

## 状態

Kaggle private CPU version 1でStage A0 raw-only censusを完了したが、
事前固定したzero-shock control well数のeligibility gateをFAILし、
`stage_a0_eligibility_failed_closed`で終了した。Stage A1、CV、Public LB、
Private LBは存在しない。

## 仮説

raw GR現在点だけが前後から孤立し、past predictive messageとcurrent observationを
除いたfuture messageが同じTVT近傍を支持する場合、current emissionを除いた
row-local posterior meanの方が親exp209 smoothed meanより安全である。

## 設定

- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Route: `pf_beam`
- scientific candidate: 1
- Stage A0: raw-only censusとtarget-free fixed64 manifest
- Stage A1予定: unchanged parent message replay 64 wells
- Stage 1上限: 別承認時のみ773 wells
- candidate state-changing HMM / parent prediction rerun /
  LightGBM / model / booster / PF / Beam / GPU: すべて0
- メトリック: unknown suffix row RMSE
- seed: RNGなし

## 結果

| メトリック | 値 |
| --- | --- |
| CV | 未実行 |
| Public LB | 未提出 |
| Private LB | 未提出 |
| Stage A0 raw census | 773 wells |
| isolated raw-shock rows | 17,047 |
| raw-shock support wells | 763 |
| zero-shock control wells | 10 / 必要32、FAIL |
| Stage A1 parent message HMM replay | 0 / 予定64 |
| candidate prediction / truth join | 未実行 |
| 専用test | 14 passed |
| exp209 / LOO 数値parity | absolute tolerance `5e-7` PASS |

## 再現性

- deterministic anchor: false
- seed policy: RNGなし、stable well/row/state/message/manifest順
- canonical train Notebook SHA256:
  `a498818082b9e29240fac8a57b06f9646433f3f0a434d18dfe51fcca1fdcfc72`
- kernel version / id_no: `1 / 129168015`
- Kaggle status: `COMPLETE`
- push / complete: `2026-07-30 12:05:48 / 12:10:10 UTC`
- log elapsed: `263.99671437 sec`
- raw census SHA256:
  `fdbb653e13bdd6132ffbe08d129fc44a744ed72b81fdc4d41ae04aa0848202cb`
- raw-shock rows decompressed SHA256:
  `1615aa3504eba71a90dc5c36f782ba79ea34162f3bd2876043b132f703332116`
- manifest / message / trigger / prediction SHA: eligibility FAILのため未生成
- model / submission SHA: 非該当
- rerun result: no-rescue契約により再実行しない

## 解釈

raw-only isolated-shock条件は763/773 wells、98.7%のwellで少なくとも1回発火し、
zero-shock wellは10しか残らなかった。このため「shock-support 32 +
zero-shock matched control 32」というtarget-free fixed64比較は実データ上で
成立しない。これはcandidate性能の否定ではなく、事前検証設計のsupport不足だが、
同一実験でthreshold、window、control定義を変えることはsame-data rescueになるため
行わない。HMM replayは0で、GPU・model・booster・PF・Beamも0だった。

## 次

exp482はterminal closeとする。threshold/window/control定義の救済、再run、
Stage A1、Stage 1、inference、submissionは行わない。次にraw shock仮説を扱うなら、
別実験の0-HMM/0-prediction preflightとして、TVTや誤差を読まずsensor-specificityを
判定できる独立raw-signal証拠を先に固定する。
