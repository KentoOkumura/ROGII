# タスクリスト

## TODO（追加の明示承認が必要）

- [x] Kaggle CPU preflight packageを作り、metadata/bootstrap configを検証する。
- [x] preflight前に2 variants × 4 wells = 8 PF well-runs、1,024 seed-well、
  512,000 particle starts、model/booster/GPU 0を再確認する。
- [x] preflight PASS後、別承認を得てfull 4 shard + mergeをpackage/pushする。
- [x] full前に1 scientific variant、773 PF well-runs、98,944 seed-well、
  49,472,000 particle starts、parent full rerun 0を再確認する。
- [x] fixed technical preflight gateを判定し、記録を更新する。
- [x] alpha0 comparatorを保存exp404 x1.0 arithmeticへ訂正したversion 3を
  実行するユーザー承認を得る。
- [x] version 3 packageを検証し、同じcanonical kernel idへpushする。
- [x] version 3 technical gateと生成物SHAを記録する。
- [x] float32 comparator復元を含むversion 4 packageを検証し、debug retryする。
- [x] version 4 technical gateと生成物SHAを記録する。

## 進行中

- なし。

## ブロック中

- なし。scientific gate FAILによりterminal close済み。

## 完了

- [x] `kaggle-review-exp`、`docs/agent-playbooks.md`、
  `docs/06_reproducibility.md`を確認した。
- [x] steeringをexperiment scaffoldより先に作成した。
- [x] `exp429_self_gr_weak_boost_likelihood_pf_ablation` scaffoldを作成した。
- [x] Route、親、PF control、self-GR式参照を固定した。
- [x] `exp091/128`をPF直接統合結果から除外した。
- [x] self-GR surface、particle log-likelihood式、state interpolationを固定した。
- [x] primary scale5 / secondary arithmetic controlとno-grid契約を固定した。
- [x] preflight/full実行量、truth-late freeze、SHA、gate、停止条件を固定した。
- [x] backlog、実験一覧、実験文書をdesign-only状態へ更新した。
- [x] design-only作成時点ではcanonical train/inference notebookを
  template placeholderのまま保持した。
- [x] ユーザーの実装承認を記録した。
- [x] exp400 compact PF kernelとexp223 self-GR surfaceの必要部分だけを抽出した
  Jupytext percent形式compact self-contained trainを実装した。
- [x] padded self-GR grid、particle-state線形補間、combined likelihood、
  ESS/resampling/seed readoutを実装した。
- [x] alpha0 parity、surface formula、truth-late freeze、execution count、
  deterministic LPT 4 shard merge、固定gateのcontract testを実装した。
- [x] target-free固定4 wells assetを作成し、SHA
  `24358da10d2d853b25b4eeb68446c005e34364c78d7f0185af4ceb601effd876`
  をconfigへ固定した。
- [x] compact trainを正規train notebookへ採用した。
- [x] compact inferenceと正規inference notebookをfail-closedで実装した。
- [x] 専用contract test 11件、構文、Ruff F821、Jupytext round-trip、
  strict experiment validationを通した。
- [x] Kaggle CPU preflight version 1のasset path failureを記録し、科学契約を
  変えずpathだけ修正したversion 2を同じcanonical kernel idで実行した。
- [x] version 2で2 variants / 4 wells / 8 PF runs / 1,024 seed-well /
  512,000 particle startsを完走し、technical FAILと生成物SHAを記録した。
- [x] preflight version 4で18,055/18,055行float32 bit-exact、
  alpha0最大差0.0 ft、technical PASSを確認した。
- [x] full 4 shardで773 wells、3,783,989 rows、98,944 seed-well、
  49,472,000 particle startsを完走した。
- [x] merge v1のmanifest dtype-only ERRORを診断し、773x5値一致を確認した。
- [x] `shard_index int8`復元fixと回帰testを追加し、13 tests、
  Jupytext、py_compile、Ruff F821、strict validationを通した。
- [x] merge version 2でtechnical gate PASS、scientific gate FAILを確定し、
  metrics、gate、manifest、SHAを保存した。
- [x] `terminal_close_without_self_gr_or_pf_rescue_grid`を記録し、
  inference / submission / same-OOF rescueへ進まないことを確認した。
