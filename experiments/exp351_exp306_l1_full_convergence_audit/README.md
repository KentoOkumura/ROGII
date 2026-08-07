# exp351_exp306_l1_full_convergence_audit

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU full audit完了、technical FAILでclosed
- CV / Public LB / Private LB: なし / なし / なし
- 作成日: 2026-07-23
- 親実験: `exp306_robust_rts_l1_convergence_calibration_audit`

## 仮説

exp306 Stage 0で唯一full-eligibleになった固定L1設定は、設定を変えず全773 wells / 1,546 seriesへ広げても全件technical PASSできる。

## 変更点

- exp306の固定64 wellsから全773 wellsへ監査範囲だけを拡張する。
- L1 `max_admm=2000, rho=1, abs/rel tol=1e-4`とlambda式を固定する。
- RTS A/Bは対象外とし、solver branchはL1の1つだけとする。
- exp306の64-well/8-well subset SHAを、新しいfull runのsubsetと完全一致させる。

## 検証方針

- 全773 wellsのhorizontal/typewell、計1,546 seriesをKaggle CPU single workerで1回実行する。
- `1,546/1,546` convergence/technical PASS、finite、length/order identity、fallback 0、error 0をAND gateとする。
- full wall timeは8.5時間以内とする。
- parent artifact SHA、raw well identity、input/output/status/parity SHAをhard gateにする。
- horizontal TVT、truth/error/formation、MRR/top3/RMSEを読まず、scientific scoreを生成しない。

## 実行量

- active branch: 1
- L1 solver: 1,546 series-runs
- Stage 0 control / full rerun / parity rerun: 0 / 0 / 0
- model / LightGBM / fold / HMM / PF / Beam / booster / control再実行 / GPU: すべて0

## 実行入口

- `exp351_exp306_l1_full_convergence_audit_compact_selfcontained_train.py` / `.ipynb`を実装候補とする。
- compact候補を正規train / inference Notebookへ採用する。
- canonical kernel `kentookumura/exp351-exp306-l1-full-convergence-audit-train` version 1 / id_no `128354027`で実行済み。
- 完了後は`run_full_l1=false`へ戻し、同一設定の再実行をfail-closeした。
- inferenceとsubmissionは本実験では常に無効とする。

## 結果

Kaggle CPU version 1を329.250秒で完了した。親anchor、raw identity、coverage、finite/order、fallback/error、64/8-well exact parity、runtimeはPASSしたが、horizontal 9 seriesがmax iteration 2000で未収束となった。convergence/technical PASSは`1,537/1,546`で、固定all-series gateを満たさないため`full_technical_fail_closed`。

## 所見

- Stage 0の128 seriesはPASSしたが、full 1,546 seriesでは9 horizontal seriesが未収束だった。
- exception、NaN、fallback、親再現性、runtimeではなく、固定2000反復以内の収束だけがFAILした。
- 設計どおりiteration/tolerance/lambda/rho/grid救済を行わず、exp304の`selected_denoiser=swt_db4_l3`は変更しない。

## 次

technical negativeとして閉じる。科学評価、inference、submissionへ進まない。
