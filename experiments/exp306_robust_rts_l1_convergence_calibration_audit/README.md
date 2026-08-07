# exp306_robust_rts_l1_convergence_calibration_audit

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU Stage 0 version 1完了・L1のみfull適格・後続full auditはexp351でdesign-only
- CV / Public LB / Private LB: なし / なし / なし
- 作成日: 2026-07-21
- 親実験: `exp304_gr_denoiser_emission_separability_readout`

## 仮説

exp304のRTS/L1 technical FAILがobjective不成立ではなく反復予算と停止許容差の不整合なら、truthやseparability scoreを使わず、事前固定した小さなsolver変更だけで全series収束を回復できる。

## 変更点

- RTS A: 最大8→32 IRLS、許容差`1e-6`は維持。
- RTS B: AがStage 0で1件でもFAILした場合だけ、最大32、許容差`1e-4`。
- L1: lambda/rho/toleranceを固定し、最大500→2000 ADMMだけ変更。
- RTS/L1を独立branchとして判定し、scientific scoreは評価しない。

## 検証方針

- Stage 0: 固定salt付きSHA256順の先頭64 wells、horizontal/typewell計128 series。
- Determinism: Stage 0先頭8 wellsのoutput/status/iteration content SHA完全一致。
- Runtime: branch別full 773-well外挿と実測が各8.5時間以内。
- Full: eligible branchごとに全773 wells / 1,546 series、全件convergence/finite/order/fallback/SHA PASSを必須とする。
- Leakage check: horizontal `TVT`、truth/error/formation、MRR/top3/RMSEを読み込まず、typewell TVTは参照series座標にだけ使う。

## 実行入口

- compact self-contained Jupytext trainを正規train Notebookへ採用済み。固定sample、RTS A→条件付きB、L1、8-well parity、runtime gate、生成物SHAをセル上で追える。
- inference Notebookは常に例外停止し、prediction/submissionを生成しない。
- Stage 0完了後は`execution.run_stage0=false`へ戻し、現configではtrain Notebookも生成物を作らず停止する。
- L1 full auditは後続`exp351_exp306_l1_full_convergence_audit`へdesign-onlyで切り出し、exp306内のfull flagはfail-closeのままとする。

## 結果

Kaggle CPU version 1（id_no `128231380`）で固定64 wells、3 branch、384 core series-runsを完了した。

- L1 `max_admm=2000`: `128/128` convergence/technical PASS、parity `16/16` exact、実測`25.161 sec`、full外挿`303.896 sec`で唯一のfull適格branch。
- RTS A `32,1e-6`: `7/128` PASS、`121/128` FAIL、実測`999.044 sec`、full外挿`12,066.577 sec`で不適格。
- 条件付きRTS B `32,1e-4`: `108/128` PASS、`20/128` FAIL、実測`695.615 sec`、full外挿`8,401.723 sec`で不適格。
- input/output/statusのraw gzip SHAと展開後SHAはgate記録値に一致し、truth/scientific scoreは未読。CV、LB、prediction、submissionは生成していない。

## 所見

- exp304のSWT選択を救済・上書きしないtechnical-only experimentとして固定した。
- RTS A→Bは条件付き1分岐でありgridではない。B FAIL後の追加調整は禁止する。
- PASSは将来の別科学実験を設計できる資格だけであり、exp306内で品質を判断しない。
- Stage 0はL1だけを技術的に支持した。RTS BはAより大幅に改善したが、horizontal 7・typewell 13 seriesが32 iterationsで未収束のため、事前gateに従いexp306内で閉じる。
- Full auditは別承認が必要。現時点で提出候補としては利用しない。

## 次

後続`exp351_exp306_l1_full_convergence_audit`の実装は別承認待ち。exp306はStage 0 evidence anchorとして固定し、科学評価、RTS追加調整、inference、submissionへ自動進行しない。
