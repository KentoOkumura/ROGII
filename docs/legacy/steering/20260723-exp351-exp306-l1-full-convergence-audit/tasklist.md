# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- scientific score、HMM/PF/Beam、inference、submissionは本実験の範囲外。

## 完了

- `exp351_exp306_l1_full_convergence_audit`を採番した。
- steeringとdesign-only experiment scaffoldを作成した。
- 親exp306 version 1のL1 Stage 0 evidence、kernel、artifact SHAを固定した。
- 対象をL1 `l1_iter2000_rho1_tol1e4`だけに限定し、RTSを除外した。
- 773 wells / 1,546 series / 1 branch / full rerun 0 / parent control再実行0を固定した。
- all-convergence、finite/order/fallback、8.5時間、64-well/8-well cross-run SHA parityのAND gateを固定した。
- model / LightGBM config / trained fold / HMM / PF / Beam / booster / GPUをすべて0に固定した。
- truth/scientific score、prediction、submissionを読まないtechnical-only境界を固定した。
- `docs/06_reproducibility.md`に従いparent artifact、input/output/status、gzip raw/decompressed SHA、Kaggle bootstrap契約を設計した。
- exp306 compact trainからtarget-free preparation、L1 solver、SHA utilityだけを抽出し、別名compact self-contained train候補を実装した。
- RTS、Stage 0分岐、scientific score、prediction、submissionを持たないfail-closed contractと別名inference候補を実装した。
- 親anchor file/content SHA、raw identity、1,546 status coverage、all-convergence、64/8-well cross-run parity、runtime、forbidden columnを監査する実装を追加した。
- parent anchor mutation、raw sample mutation、1-series failure、parity mutation、runtime超過、forbidden columnを含む11件のsynthetic testsを追加した。
- exp306親compact 11章・1,400行に対し、exp351 compact trainは10章・1,461行で、親anchor guard/full gate/生成物保存をNotebook上で追えることを確認した。
- Jupytext candidate Notebook生成/round-trip、py_compile、Ruff F821、専用11 tests、関連22 tests、strict experiment validation、template validationをPASSした。
- compact候補を正規train/inference Notebookへ採用し、cell type/source完全一致を確認した。
- Kaggle CPU version 1 / id_no `128354027`を1 branch / 1,546 series-runs / booster 0で完了した。
- 2026-07-23の追加依頼で正規Notebook採用とKaggle CPU full audit 1回のpackage/push/run承認を得た。
- push前実行量を1 branch / 1,546 solver series-runs / model・LightGBM・fold・HMM・PF・Beam・booster・control再実行・GPU各0と再確認した。
- metadata/bootstrap内のparent anchor、L1設定、run flag、CPU/internet/thread設定を照合した。
- 親anchor、raw identity、coverage、finite/order、fallback/error、64/8-well exact parity、runtimeをPASSした。
- horizontal 9 seriesがmax iteration 2000で未収束となり、`1,537/1,546` convergence/technical PASSで固定all-series gateをFAILした。
- `full_technical_fail_closed`としてiteration/tolerance/lambda/rho/grid救済なしで閉じた。
- input/output/statusのcontent/raw/decompressed SHAをlogsと選択取得したgate/statusで記録した。
- scientific score、HMM/PF/Beam、inference、submissionは実行していない。
