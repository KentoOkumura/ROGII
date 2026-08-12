# タスクリスト

## 目的

固定U境界fadeをtarget-freeかつno-rescueで監査できる状態までの作業を段階管理し、design-only作成から実装・実行へ自動移行しない。

## 未実行（terminal close）

- raw-test inference、submission、Public LB probeは、Stage 0 pooled gain gate FAILにより実行しない。
- cap/tau/threshold/distance/well gate/blend/親/gate変更による同一OOF救済は行わない。

## 完了

- `exp349_exp287_u_boundary_continuity_fade`として採番した。
- `docs/legacy/steering/20260722-exp349-exp287-u-boundary-continuity-fade/`を作成した。
- 親をexp287、routeを`ml_model`、変更を`cap=8.0 / tau=240.0`の単一U境界fadeへ固定した。
- target-free generation／SHA freeze／late-truth join、technical/scientific AND gate、failure policyを固定した。
- 1 postprocess variant、0 model／0 booster／0 GPU／0 control再学習の実行契約を記録した。
- experiment scaffoldを作成し、design-only／未実装へ固定した。
- ユーザーの明示依頼によりimplementationとcompact self-contained Notebook候補を承認済みに更新した。
- exp287 OOF pretruth projection、raw prefix/suffix、固定U-fade、candidate/diagnostic SHA freeze、late-truth join、全technical/scientific gateをcompact train候補へ実装した。
- inference候補をStage 0 PASS前のprediction/submission禁止のfail-closed entrypointとして実装した。
- prefix/suffix、formula/cap/fade、identity、禁止列、freeze改ざん、fixed gateを検証するsynthetic tests 10件を追加した。
- compact train/inferenceのJupytext sourceと`.ipynb`候補を生成した。既存canonical Notebookは上書きしていない。
- 1 variant、5 reporting folds、trained fold/model/config/booster/PF/Beam/HMM/control再学習/GPUが`0/0/0/0/0/0/0/0/0`であることを再確認した。
- Jupytext往復、py_compile、Ruff、専用pytest 10件、strict experiment validation、experiment docs reviewをPASSした。
- ユーザー承認後、compact候補を正規train Notebookへ採用し、exp287 OOF/model manifest、competition raw train、固定hidden-like assignmentをKaggle入力上でpreflightした。
- private CPU packageをcanonical kernelへpushした。version 1はpandas返り値型互換でfreeze前にtechnical errorとなり、仮説・入力・gateを変えない型修正だけでversion 2を実行した。
- version 2は3,783,989 rows / 773 wells、1 variant / 5 reporting folds / model・booster・GPU 0を約156秒で完了した。
- 全technical gateとscientific gate 11/12件をPASS。親`8.136708220`から候補`8.135096925`へ`0.001611295 ft`改善したが、pooled下限`0.020 ft`をFAILした。
- 0--240は`0.110003778 ft`、5/5 folds、hidden-like 2面、far／by-well safetyはPASSした。`FAIL_CLOSE_NO_RESCUE`としてdirect fixed U-boundary fadeを閉じた。
- logsと選択取得した小規模metrics/manifestsからruntime、生成物path、SHAを記録し、取得物のSHA一致を確認した。

## 次のアクション

- exp349内の追加実行はない。continuity再訪は独立したtarget-free feature／selector仮説を事前設計できた場合だけ別途判断する。
