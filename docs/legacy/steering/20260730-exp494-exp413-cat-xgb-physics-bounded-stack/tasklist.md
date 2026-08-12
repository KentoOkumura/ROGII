# タスクリスト

## TODO

- なし。

## ブロック中

- conditional gateとsame-OOF rescueは引き続き禁止。
- scientific anchor昇格は固定tail gate FAILにより禁止。

## 完了

- ユーザーがscientific FAILを保持した参考提出overrideを明示した。
- constant stackのweight、保存model、禁止する追加処理を再凍結した。
- compact self-contained inferenceを正規Notebookへ反映した。
- Kaggle private T4 / internet offでhidden-safe inferenceを完了した。
- outputのroot `submission.csv`をsample submissionに対して検証し、
  FAIL 0 / WARN 0を確認した。
- constant stackをref `55134873`として参考提出し、268分の監視後に
  `COMPLETE` / Public LB `7.228`を確認した。
- exp413 Public LB `7.201`比`+0.027`悪化のためexp494を不採用とし、
  CV / LB / SHA / override / anchor扱いを全記録へ反映した。

- backlog候補とexp494番号を確定した。
- 実験scaffoldを作成した。
- requirements、design、tasklistを作成した。
- Stage 0--6の入力、出力、停止条件を固定した。
- exp413 final370 / fold / OOF / saved model freeze契約を固定した。
- CatBoost / XGBoostのconfig、fold、合計10 modelsを固定した。
- 物理候補を`exp226_w500_50_50` 1本へ固定した。
- bounded stack、採用gate、conditional confidence gateを固定した。
- hidden-safe dynamic cardinalityと9時間runtime契約を固定した。
- canonical notebookをmarkdown-only placeholderにする方針を固定した。
- ユーザーの実装依頼を受け、Stage 0 final370 freezeをJupytext percent形式の
  別名train候補へ実装した。
- exp413 Stage 0/C/S/DのSHA、row key、fold、370列、float32 matrix content SHAの
  fail-closed parity guardを実装した。
- CatBoost `cb0` 5 modelsとXGBoost Cdeotte v3 5 boostersだけを学習する
  fold orchestrationを実装した。
- family別OOF / correlation / residual correlation / covariance /
  disagreement / scope / hidden-like / by-well監査を実装した。
- `exp226_w500_50_50`だけを読むphysical candidate guardを実装した。
- fixed-bound SLSQPの5-fold OOF-level cross-fitとdeployment bounded-simplex
  projectionを実装した。
- constant stack PASS時だけ動く0.25 ft capのsmall disagreement gateを実装した。
- 親exp413 compact trainの9章766行に対し、exp494候補は9章2258行で、
  入力凍結、2-family学習、4-family監査、stack/gateを追加していることを確認した。
- Jupytext変換/test、`py_compile`、`ruff --select F821`、10 unit tests、
  `make validate-exp`を通過した。
- 2026-07-31のユーザー実行依頼をcanonical train採用、Kaggle package /
  push / Stage 0--5 train runの明示承認として記録した。
- push前の実行量が2 variants / 2 configs / 5 folds / 10 GPU models、
  control・selector・PF/HMM/Beam再学習0であることを再確認した。
- compact候補をcanonical train Notebookへ採用し、private T4 /
  internet off / run-on-push packageを生成した。
- canonical kernel version 1（id_no `129213293`）をpushして実行開始し、
  metadata pullで`NvidiaTeslaT4`の反映を確認した。
- version 1は`ReplacementCandidateCache`が同名候補へscale5 overlayを
  適用した実測RMSE `8.0702187939`と、凍結exp263 RMSE `8.238331`の不一致を
  Stage 0で検出し、学習0本・CVなしでfail closedした。
- ユーザー確認により親exp413とのtrain/inference parityを優先し、
  `likpf_mean=likpf_scale_5_x1p0`のscale5-overlay版へ契約を一意化した。
  exp263同名候補のOOF/Public LB根拠はscale5版へ転用しない。
- 更新契約の専用test / Jupytext / 構文 / F821 / strict validation /
  package config parityを通過し、同じcanonical kernelへversion 2をpushした。
- version 2は約1395秒で`DeadKernelError`。fold完了log 0、Kaggle files 0、
  CVなしで停止した。
- full matrix SHAのzero-copy化、CatBoost Pool後の生行列解放、
  CatBoost/XGBoost fold matrixの直列化と進捗logを実装した。
- version 3も2 variants / 2 configs / 5 folds / 10 GPU models、
  control・selector・PF/HMM/Beam再学習0であることを再確認した。
- 更新後のJupytext / 構文 / F821 / 専用11 testsを通過した。
- private T4 / internet off packageを検証し、同じcanonical kernelへ
  version 3をpushした。push後pullでもmemory fixとT4反映を確認した。
- version 3はStage 0 matrix preflight 5/5完了後、family train開始前の
  `DeadKernelError`で停止した。完了model / CV / reusable outputは0。
- allocator trim、25万行chunk物理OOF Parquet、列先行matrix assembly、
  chunk finite検証、RSS進捗logを実装した。
- version 4も2 variants / 2 configs / 5 folds / 10 GPU models、
  control・selector・PF/HMM/Beam再学習0であることを再確認した。
- 更新後のJupytext / 構文 / F821 / 専用12 testsを通過した。
- private T4 / internet off packageを検証し、同じcanonical kernelへ
  version 4をpushした。push後pullでもStage 0 memory fixとT4反映を確認した。
- version 4はStage 0とfold 0 CatBoost Pool生成を完了した後、fit開始時の
  `DeadKernelError`で停止した。完了model / CV / reusable outputは0。
- clean273 273特徴の一時float32 NPY memmap化、学習前DataFrame解放、
  memmap fold assemblyの凍結SHA再検算、CatBoost train / valid raw matrixの
  直列解放を実装した。
- version 5も2 variants / 2 configs / 5 folds / 10 GPU models、
  control・selector・PF/HMM/Beam再学習0であることを再確認した。
- 更新後のJupytext / 構文 / F821 / 専用13 tests、
  strict experiment / template validationを通過した。
- private T4 / internet off packageを検証し、同じcanonical kernelへ
  version 5をpushした。remote 21 cell source、embedded config SHA、
  memmap fix、CatBoost raw matrix直列解放、T4反映を確認した。
- version 5は`KernelWorkerStatus.COMPLETE`、`5187.904674 sec`で
  CatBoost 5 + XGBoost 5の10/10 modelsを完走した。control / selector /
  PF/HMM/Beam再学習は0。
- bounded stackはexp413 `7.884802794`から`7.827450885`へ
  `0.057351909 ft`改善し、5/5 folds、全固定scope、hidden-like 2面を改善した。
- by-well p95 `+0.634420635 ft`、worst `+3.843640672 ft`で固定tail
  2条件をFAILしたため、`train_complete_guard_failed_closed`として
  exp413を維持し、conditional gate / inference / submissionへ進まない。
- selected metrics / fold / scope / hidden / weight / model /
  reproducibility manifestだけを`kaggle/output/train_v5_selected/`へ保存した。
- model manifest、OOF、weight、reproducibility SHAを記録した。
  reproducibility manifest内の最終metrics SHAだけはpost-manifest rewriteにより
  staleだが、最終metricsとmanifestの相互参照および他のselective SHAは一致した。
- KAGGLE_DIRECTIONの実装済みexp494行をbacklogから削除し、terminal resultを
  完了記録へ移した。
