# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- 正規Notebook採用、competition submissionは未承認。

## 完了

- Stage D v4でRidge中間物解放、shared PF/SP45の4-thread bounded streaming、SP45→HJYACT→exp413の
  DataFrame ownership transfer、hidden-safe visible SHA guardを実装した。
- Stage D v4 source / Notebookを再生成し、構文、Ruff F821、Jupytext round-trip、19 contract tests、
  strict experiment validationをPASSした。competition再提出は未承認。
- Stage D v4 packageを`--no-src`で再生成し、26 support files、private T4、internet off、remote/local
  56 cell source完全一致を確認した。同じcanonical kernel version 4でvisible実行を完走した。
- v2の5出力SHA完全一致、submission check PASS、Ridge/exp413前/shared-bank終了時のcurrent RSS低下、
  200-well推定`6.289658〜8.057147時間`を確認した。hidden OOM保証とは扱わず、再提出は行っていない。

- exp512とexp413 likelihood-PFのsource差、consumer、visible override、runtime外挿を監査した。
- 共有する5手順、共有禁止component、stable seed、memory lifetimeを設計として固定した。
- fixed32 technical、fixed32 paired精度screening、Stage D visible readinessの段階gateを固定した。
- model config 0、fold training 0、booster 0、親/control再学習0を固定した。
- backlog、steering、実験scaffoldを作成し、設計確定・実装未承認として記録した。
- ユーザーの`exp514を実装してください`を実装承認として記録し、親exp512、exp073 replay、
  exp413 configの設計時SHAがdriftしていないことを確認した。
- SHA固定した親exp512候補を変換する`prepare_exp514_shared_likpf_candidate.py`を作成した。
- exp073/exp413とAST一致するstable seed / Numba core、well内だけに保持するraw bank、
  scale 3/5/8/12・mean・branch summary、SP45 / exp413 adapterを実装した。
- SP45 legacy bankとexp413後段`build_likpf`を実行経路から外し、consumer各1回、fallback 0を
  generation ledgerでfail-closeするようにした。
- compact self-contained inference候補と、truth/submissionを読まないStage A fixed32専用Notebookを生成した。
- dedicated contract test 7件、構文、Ruff F821、Jupytext round-trip、strict `validate-exp`をPASSした。
- 正規train / inference Notebookはplaceholderのまま保持した。
- Stage A専用NotebookをKaggle private T4 version 1で完走し、fixed32 4 runのaggregate / branch /
  ledger SHA完全一致、truth read 0、booster 0、親再学習0、submission 0を確認してtechnical gateをPASSした。
- ユーザー指示でStage Bを200 wellsからStage Aと同じ固定32 wellsへ縮小し、selection再選択禁止、
  閾値維持、small-screening解釈、Stage C 200-well維持を設計へ反映した。Stage B実行は未承認。
- ユーザーのStage B実行指示を受け、Stage B fixed32の実装/package/run/report取得を承認済みとした。
- legacy/shared各32 bank、common Beam、truth freeze後join、親selector/branch hedge固定、全AND gateを持つ
  Stage B専用Jupytext source / Notebookとcontract testsを実装した。
- ユーザー指示によりStage C 200-well shadowを不要化した。Stage CはPASSではなく
  `not_required_by_user_override`として記録する。
- 200-well runtimeはStage D visible testの工程別時間から、4-way並列工程と逐次工程を分けて外挿する。
  これはhidden runtime実測や9時間完走保証ではない。
- Stage B v1は129,906行のprediction freezeまで`5,081.673秒`で完了したが、pre-branch採点の
  同名列衝突でERROR。scientific gateは未評価として記録した。
- Stage D visible testの実装/package/run/output取得をユーザー承認済みとし、正規Notebookを上書きせず
  別名source / Notebookを生成した。
- Stage D version 1をprivate T4 / internet offで開始した。
- ユーザー指示でStage B修正・再実行を承認済みとし、評価関数だけをcopy + 1次元配列代入へ修正した。
- Stage Bの静的/contract/strict検証をPASSし、同じcanonical kernelへversion 2をpushした。
- Stage D v1は3 wells / 14,151行の部分推論後、科学変更と両立しない親visible exact SHA guardでERROR。
  runtime reportと最終50/50 submissionには未到達として記録した。
- v1 partial + 親exp413 proxyの200-well暫定推定を`8.448〜9.831時間`、保守推定を
  `9.129〜10.739時間`として記録し、上限基準で`estimated_fail`とした。
- 親exact SHA guardをv1 exp514 candidate witnessへ置換した。科学条件とruntime式は不変。
- Stage D v2の静的/contract/strict検証をPASSし、同じcanonical kernelへversion 2をpushした。
- Stage D v2は3 wells / 14,151行を929.929790秒で完走し、200-well推定を
  `8.068150〜9.528814時間`、上限基準`estimated_fail`として記録した。
- Gold 4-process、SP45/HJYACT決定論feature共有、v2の5出力SHA fail-close guardを実装した。
- Stage D v3の構文、Ruff F821、Jupytext、16 contract tests、strict validation、package作成をPASSした。
- Stage D v3を同じcanonical kernel version 3で完走し、5出力SHA完全一致、Gold effective 3 process、
  HJYACT決定論176列共有、submission check PASSを確認した。
- visible全体をv2比90.233199秒短縮し、200-well推定を`6.174531〜7.957332時間`、
  `estimated_pass_not_hidden_runtime_guarantee`として記録した。
- Stage B v2は32 wells / 129,906 rowsを`4,845.475189秒`で正常完走した。v1とprediction SHAが一致し、
  採点修正だけであることを確認した。
- primary pooled delta `+0.049680 ft`、nonworse fold `2/5`、raw-GR observed `+0.060618 ft`、
  by-well p95 `+0.647871 ft`で固定all-ANDをFAILした。事前規約どおり救済せずexp514を終端した。
- `SESSION_NOTES.md`、`result.md`、`metrics.json`、`config.yaml`、`experiment_summary.md`、
  `KAGGLE_DIRECTION.md`へ終端結果を反映し、Stage D submissionを提出不可とした。
- ユーザー実施のStage D v3 code submission ref `55266559`はhidden rerun unhandled errorでscoreなし。
  visible v2固定SHAの無条件guardというhidden-incompatible defectを記録し、再提出なしで終端した。
