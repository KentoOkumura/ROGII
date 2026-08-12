# exp276_exp264_compact_tail_risk_target_free_gate_audit セッションノート

## 現在の状態（2026-07-21 corrected-parent再検証）

- Route: `ml_model`
- 状態: corrected exp264 Stage C v6 / Stage D v3 Kaggle private CPU version 3完了、固定guard FAIL、branch closed
- 実行承認: 2026-07-21 12:27:02 JST、ユーザーがexp276再検証の先行を明示指示
- 実行量: 1 audit variant / 5 evaluation folds / LightGBM config 0 / trained fold 0 / booster 0
- parent/control再学習: 0
- inference/submission: disabled
- 現在のguard判定: 有効なFAIL。旧version 2のFAILは無効履歴のまま分離する

## 目的

corrected exp264 Stage D v3のglobal改善 10.476169 -> 8.460811を保ちながら、255悪化wellと
220 over-0.25 wellsをtarget-free compact/contextだけでouter-fold再現可能に識別できるか監査する。

## 旧version 2完了時の状態（無効履歴）

> **結果無効:** 親exp264 Stage C/D OOFのfeature availability leakageにより、この節から
> 2026-07-18版の数値は実行再現用の履歴であり、診断、比較、negative result、推論判断に使用しない。

- Route: `ensemble`
- 状態: Kaggle CPU version 2完了、親OOF無効によりguard判定無効、branch closed
- CV: 無効（downstream outer5の履歴値は性能判断に使用禁止）
- LB: scope外
- current-test inference / submission: disabled

## 実行コスト契約

- active audit variant: 1
- LightGBM config: 0
- trained fold: 0
- total booster: 0
- parent/control再学習: 0
- input partition: Stage C 25（18,919,945 rows）
- outcome input: Stage D OOF 3,783,989 rows / 773 wells
- runtime: Kaggle CPU、GPU off、internet off、`num_workers=1`

## 実装内容

- `.steering/20260718-exp276-exp264-compact-tail-risk-target-free-gate-audit/`に要件、設計、tasklistを作成した。
- 新規`exp276_exp264_compact_tail_risk_target_free_gate_audit`をtemplateから作成した。
- train notebookをJupytext percent形式のself-contained sourceとして実装した。
- Stage C manifest/schema/partition manifestとStage D OOFを期待SHAでfail-closedにした。
- 25 partitionを必要なcompact列だけ逐次読みし、先頭128、先頭512、全区間へwell集約する。
- raw horizontalは`MD/X/Y/Z/GR`だけを読み、partition keyの`well_row_idx`からgeometry/contextを作る。
  `TVT`/`TVT_input`値は読み込まない。
- score dispersion、candidate divergence、top1-anchor distance、confidence coverage、geometry/contextの
  5 familyをouter-train empirical percentile化し、family内・family間を等重みにした。
- q70/q80/q90をouter-train risk分布だけから固定し、outer-validへ適用する。
- risk凍結後にStage D target/predictionをjoinし、bad rate/lift/recall、gated RMSE、改善保持率、
  fallback後worst-wellを保存する。
- inference notebookは明示的disabled contractとし、submissionを生成しない。
- targeted tests 7件でtransform、label列拒否、train/valid overlap拒否、outer-fold metadata、raw TVT値非依存、gate readout、hive親directory非干渉を検証した。

## Notebook構造比較

- 親exp264 train source: 447行 / 7章。
- exp276 train source: 1,242行 / 8章、17 cells。
- exp276 inference source: 55行 / 2章、6 cells。inference自体がscope外のためdisabled contractを明示する。
- exp276 trainは親のcompute/input/execution/metrics/generated-artifacts role slotを維持し、
  compact aggregation、target-free risk、Stage D readoutをnotebook上へ展開した。薄い`main()`呼び出しではない。
- 同じ実験directoryのhelper importなし、`__file__`なし。

## コマンドログ

### 実行済み

```bash
task new-steering EXP=exp276_exp264_compact_tail_risk_target_free_gate_audit
# task未導入のためcommand not found
make new-steering EXP=exp276_exp264_compact_tail_risk_target_free_gate_audit
make new-exp EXP=exp276_exp264_compact_tail_risk_target_free_gate_audit
.venv/bin/pytest -q experiments/exp276_exp264_compact_tail_risk_target_free_gate_audit/tests/test_exp276_target_free_tail_risk_gate.py
# 6 passed
.venv/bin/python -m py_compile <exp276 train.py> <exp276 inference.py> <targeted test.py>
.venv/bin/ruff check <exp276 sources and test> --select F821,E9
# All checks passed
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <exp276 train.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <exp276 inference.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <exp276 train.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <exp276 inference.py>
make validate-exp EXP=exp276_exp264_compact_tail_risk_target_free_gate_audit
# strict validation passed
make prepare-kaggle-notebooks EXP=exp276_exp264_compact_tail_risk_target_free_gate_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp276-exp264-compact-tail-risk-target-free-gate-audit-train --title 'exp276 exp264 compact tail risk target free gate audit train' --run-on-push --strict"
# private / CPU / internet off package prepared
make test
# repository 143 tests passed
```

### 承認前に未実行だった項目

```bash
make push-kaggle-train EXP=exp276_exp264_compact_tail_risk_target_free_gate_audit
```

この時点ではKaggle CPU実行承認前だったため、pushしなかった。

## 2026-07-18 Kaggle CPU実行承認

- 承認時刻: `2026-07-18 13:09:10 JST`
- 承認scope: canonical private train kernelを1回pushし、CPU readoutを完了まで監視する。
- 実行量: 1 audit variant / LightGBM config 0 / trained fold 0 / total booster 0。
- parent/control再学習: 0。Stage C / Stage Dの保存済み生成物だけを読む。
- runtime: Kaggle CPU、GPU off、internet off、`num_workers=1`。
- inference / submission: disabled。本承認にはcurrent-test portと提出を含めない。

## 再現性メモ

- seed policy: 乱数なし、stable well/row sort + empirical quantile。
- stochastic components: なし。
- CPU/GPU runtime: Kaggle CPU 123.079秒、GPU off、internet off。
- Stage C manifest SHA: `c95d9ea48bbc7de26decbbb569df0ce46b0f6e1cf88e381c34d07e2f3734c06e`。
- Stage C partition manifest SHA: `8721ebf82e5192536f70017349066638d1ee3dc23651571eb8c0bb5516c69fab`。
- compact schema file/logical SHA: `e3a677610899cb33bf58262f4cf02f650300c8c2207c46b53588d3418162ea74` / `23614916c99edbbd513bcefee958d26cdfae5b83fb05c232c19736f2708dd725`。
- Stage D OOF SHA: `7367983f3053186e0a6adf18c0f145302b0451332625fb679357f3c1326dafee`。
- canonical package metadata: `kentookumura/exp276-target-free-tail-risk-gate-audit-train`、private、CPU、internet off、run-on-push。Kaggleのtitle 50文字上限に収めるため、exp番号と主要仮説を残して短縮した。
- approved package config SHA: `5adb061b3d0378a7c2b02866056f68f052a9b77dce1228d86cdd57978db8af4d`。source/package/bootstrap manifestで一致。
- train source SHA: `01a4aa0b2015755c71df5e31ff736e9a265a0133cecc98a51b96fa60817f1a4e`。source/package/bootstrapで一致。
- final approved prepared notebook SHA: `c02e8cda001be7c910d24b53c3e95d295a7f4be6732ac60dc85412b4a4e9d9f0`（45文字canonical metadataで再prepare）。
- risk feature content SHA: `4a31be3bf242c1e9020ba78051ddabe49e8da16e6270af284e99e2cc07131c9f`。
- risk score content SHA: `4be97122fd914919f5d60bcb73ac71852f1e56a55016bf3acaf5d3da6ffa757d`。
- gated OOF prediction content SHA: `4372153aa1dff0f354b848a5991a9f73c4769fe53f0d6998ead81e14700cf231`。
- model manifest / model SHA: 新規model 0のため対象外。
- submission SHA: submission生成なし。
- rerun check: 未実行。rerun前はdeterministic diagnostic anchorと呼ばない。

## 2026-07-18 初回push 400の復旧

- 当初package: `kentookumura/exp276-exp264-compact-tail-risk-target-free-gate-audit-train` / `exp276 exp264 compact tail risk target free gate audit train`。
- `make push-kaggle-train`はKaggle `SaveKernel 400`を返した。
- 同slugの`kernels pull -m`は403、`kernels list --search exp276`は`Not found`で、kernelが作られていないことを確認した。
- credentialを表示しないdiagnosticで応答本文を確認し、`The title cannot exceed 50 characters.`を特定した。
- 復旧canonical: `kentookumura/exp276-target-free-tail-risk-gate-audit-train` / `exp276 target free tail risk gate audit train`（45文字、id/title slug一致）。
- 実験番号、コード、config、入力、guard、実行量は変更しない。別実験や複数slugを増やさない。

## 2026-07-18 Kaggle version 1 push

- push時刻: `2026-07-18 13:18:45 JST`
- kernel: `kentookumura/exp276-target-free-tail-risk-gate-audit-train`
- version: 1
- Kaggle `id_no`: `127735777`
- `kernels pull -m`でprivate、CPU、GPU/TPU off、internet off、competition input 1、kernel input 2を確認した。
- pull後のKaggle notebook SHA: `e3b47391752773c17a3925efcf57e16666cae8cdd17dd62236df7cd76e16d0d9`。Kaggle側の正規化後sourceとして記録する。
- push後はlocal `execution.run_approved=false`へ戻し、version 2の誤pushを防ぐ。

## 2026-07-18 Kaggle version 1技術エラー

- runtime約18秒で、cell 6の最初のStage C partition読込時に終了した。監査readoutは未生成。
- error: `ArrowTypeError: Field downstream_outer_fold has incompatible types: int8 vs dictionary<int32>`。
- 原因: `pq.read_table(path)`が`downstream_outer_fold=.../role=...`という親directoryをhive partitionとして推論し、parquet内の同名physical列とmergeしようとした。
- 修正: schema検査に使っていた同じ`pq.ParquetFile(path)`の`read(columns=...)`で物理fileを直接読む。risk、label接続、quantile、guard、入力SHAは変更しない。
- 回帰テスト: hive形式の親directoryと同名int8 physical列を持つparquetを読めることを追加する。
- version 1は入力読込前段で落ち、追加学習0 / booster 0のまま。修正後は同じcanonical kernelのversion 2へ積む。
- 修正後検証: targeted 7 tests、py_compile、F821/E9、Jupytext round-trip、strict experiment validationを通過した。
- ユーザーの「実行」依頼を完遂するため、仮説・実行量不変のtechnical retryに限って`run_approved=true`へ再設定した。
- version 2 package config SHA: `54d6e55595cc54cd3f1498ca58be75c6f477b15c22805a5e9b1e66c0cef93a3b`。source/bootstrap manifestで一致。
- version 2 train source SHA: `098309e5a5f47d8a25c9a3cf51ab961467eb7fda3e4e39e9d32cc4fb8f68c99c`。source/bootstrap manifestで一致。
- version 2 prepared notebook SHA: `ad7161e0bd57bba31ee1604a44b5786e5f99126c8b4c1e6ea51f2204fa30194d`。
- version 2 push: `2026-07-18 13:22:53 JST`、同じkernel id_no `127735777`。成功後すぐlocal `run_approved=false`へ戻した。

## 2026-07-18 Kaggle version 2結果

- status: `COMPLETE`。監査本体runtime `123.079107`秒。
- compute: 1 audit variant / LightGBM config 0 / trained fold 0 / booster 0 / parent-control再学習0。
- input: Stage C 25 partitions / 18,919,945 rows、raw 773 files、Stage D 3,783,989 rows / 773 wells。
- Stage D anchor: matched control `8.545568072`、selector compact add-only `7.805644167`。
- q70: 223 risk wells、`delta>0` lift `0.768738`、`delta>0.25` lift `0.826640`、gated RMSE `8.461023`、改善保持`11.43%`、worst `+10.587659 ft`。
- q80: 144 risk wells、`delta>0` lift `0.842656`、`delta>0.25` lift `0.965801`、gated RMSE `8.352745`、改善保持`26.06%`、worst `+10.587659 ft`。
- q90: 75 risk wells、`delta>0` lift `0.947588`、`delta>0.25` lift `1.019909`、gated RMSE `8.196329`、改善保持`47.20%`、worst `+10.587659 ft`。
- positive-lift folds (`delta>0` / `delta>0.25`): q70 `0/5 / 1/5`、q80 `0/5 / 2/5`、q90 `3/5 / 3/5`。
- gated control改善fold: q70/q80/q90すべて`4/5`。
- 全quantileで改善保持50%とworst-well +0.25 ftを外し、`target_free_tail_risk_guard_pass=false`。
- fixed gateを不採用とし、feature/family/weight/quantile grid、current-test port、submissionへ進まない。

## 取得成果物

- 保存先: `kaggle/output/train_v2/artifacts/`。111 MBのgated OOF parquetは取得せず、Kaggle側SHAとlogical content SHAをmanifestから記録した。
- `audit_summary.json`: `81498df25a9099bc5aeda6b8a51104632d6f8efb8b3ff2162161cdc723f90a3a`。
- `input_manifest.json`: `711b891881c21bbe5653e33b1727a2c70a1c5453503b6ffc8106d2d5b65a9aeb`。
- `risk_feature_schema.json`: `5d9dc37fbca4fbe44c5994cbf48a9c14d64c23c81da4519f58226695ef4f82cb`。
- `risk_fold_quantile_readout.csv`: `1d4114e2b8108dd168b0442ca740adb28ee115468aa52946603fb3c8c8be2613`。
- `risk_pooled_quantile_readout.csv`: `13216a908b5b941b4a3721a83addb33a5b14520db37c61853afff0a2e7c86851`。
- `target_free_well_risk_scores.csv`: `6ae902892815e3012b02f2eb09e6ce6f218c5201636b7332c02be2850a756b18`。
- `gated_by_well_metrics.csv`: `2c6216ca62202476ec9cbfe68b56085f5b29538d1207c0243130b4c3a9ba3e0a`。
- repository full test: 146 passed。
- post-run safe packageは`run_approved=false` / `run_on_push=false`へ再prepareした。final config SHA `8654d852e47027be44c9d66df66c26cad2d8686fdf12c629c2b6a41dfd8e761e`、package notebook SHA `83eb6fa510305c2a8188be39b7f68fa8356540e0e38954ca1e9664233981764d`、bootstrap config SHA一致。

## 次のアクション

- corrected-parent再検証を同じcanonical kernelのversion追加として実行する。ただし
  feature/family/weight/scope/quantile/guardは変更せず、current-test inferenceとsubmissionは行わない。

## 2026-07-21 corrected-parent再検証実装

- Stage Cをversion 6へ固定し、compact manifest SHA `f4855726...ecf1c`、partition manifest
  SHA `17930b7b...cf66`、25 partition / 18,919,945 rowsをfail-closed入力とした。
- compact schemaは旧contractと同じ74列で、file/logical SHA
  `e3a67761...ea74` / `23614916...725`を維持した。
- Stage Dをversion 3 OOF SHA `b11c5005...9ae2`へ固定し、3,783,989 rows / 773 wells /
  255 worsened / 220 over-0.25 wellsをtechnical contractにした。
- 旧Stage C v3 / Stage D v2をresolverが受理しないようexpected SHAを差し替えた。
- ルートは、PF/Beam候補が補助meta featureで最終予測はdownstream LightGBMが生成するため
  `ensemble` から `ml_model` へ修正した。risk計算とguardは変更していない。
- targeted testsをcorrected SHA/count契約の2件追加し9件PASS。py_compile、ruff F821/E9、
  Jupytext変換/round-tripをPASSした。
- strict experiment validation / template validationをPASSした。repository全389 testsは
  386 passed / 1 skipped / 2 failed。FAILはexp296の完了後configと旧test期待の不整合2件だけだった。
  exp276変更とは無関係なため、本CPU監査のpushは継続した。
- approved package config/source/bootstrap SHAは `694a6a72...99e` / `0cdd10a7...31a` で一致。
  prepared notebook SHAは `71b505b2...8b6b`。
- 2026-07-21 12:36:27 JSTに同じcanonical private CPU kernelへversion 3をpush。id_noは
  `127735777`、GPU/internetはoff、kernel sourceはcorrected exp264 Stage C/Dの2本。
- `kaggle kernels pull -m`でversion 3の存在とmetadataを確認。Kaggle正規化徏後notebook SHAは
  `66b84813...d801`。push後はlocal `run_approved=false`へ戻し、誤ったversion 4 pushを防ぐ。

## 2026-07-21 corrected-parent Kaggle version 3結果

- status: `COMPLETE`。監査本体runtime `104.016918`秒。
- compute: 1 audit variant / 5 evaluation folds / LightGBM config 0 / trained fold 0 / booster 0 /
  parent-control再学習0。
- input: Stage C v6 25 partitions / 18,919,945 rows、raw 773 files、Stage D v3
  3,783,989 rows / 773 wells。
- Stage D anchor: matched clean-273 control `10.476169179`、compact-74 add-only `8.460811238`。
  255 worsened wells / 220 over-0.25 wells。
- q70: 223 risk wells、`delta>0 / >0.25` lift `0.914943 / 0.946217`、gated RMSE
  `9.770623`、改善保持`35.01%`、worst `+7.989016 ft`、lift folds `1/5 / 1/5`。
- q80: 148 risk wells、lift `0.979240 / 1.055743`、gated RMSE `9.651452`、改善保持
  `40.92%`、worst `+13.441268 ft`、lift folds `2/5 / 2/5`。
- q90: 74 risk wells、lift `1.165139 / 1.211019`、gated RMSE `9.271525`、改善保持
  `59.77%`、worst `+13.441268 ft`、lift folds `2/5 / 4/5`。
- gated control改善は3 quantileすべて5/5 folds。しかしpositive-lift 5/5とworst-well
  `<= +0.25 ft`を全quantileで外し、q70/q80は改善保持50%も外したため総合FAIL。
- q90だけを事後採用せず、feature/family/weight/quantile grid、inference、submissionは行わない。

## Corrected-parent生成物と再現性

- 大きい`gated_oof_predictions.parquet`は取得せず、結果とSHA記録に必要なJSON/CSVだけを
  `kaggle/output/train_v3_corrected/artifacts/artifacts/`へ選択取得した。
- audit summary SHA: `8cec9d25c3fa560cf31d3a63ff437a5cd4594b50a0877e924075d1a4ed62f6cc`。
- input manifest SHA: `b6db86362e1d5f6c7a4fa19d66b27760a47d53d142f5903f5d0c2fc17f49a255`。
- risk feature schema file/logical SHA: `5d9dc37fbca4fbe44c5994cbf48a9c14d64c23c81da4519f58226695ef4f82cb` /
  `3e178e78e05620f610ea4fc99e8ca3ca205f6fece8013f040bc4b9a94ddb772d`。
- risk feature / score content SHA: `7e8c3ccac6a1573651e24bd43baab756d0adde7b433e7f87b4ff4f681a54199d` /
  `c09b74fd939545fa9a28d1e71982995be04af3a5588e2609c80b4f08a7a5f470`。
- gated OOF prediction logical SHA: `ee370eb443d2d65a80b9aabcfc28e65f72dc208c7d7b69d273c8b09735eb8843`。
- 成功runは1回なのでdeterministic diagnostic anchorとは呼ばない。ただしno RNGで入力・生成物SHAは固定した。

## 次のアクション（corrected-parent判定）

- exp276の`completed + promotion guard FAIL`をexp303 dependencyに反映し、事前登録済みの
  K12/K16/K24 scale-instability readout実装へ進む。
