# exp349_exp287_u_boundary_continuity_fade セッションノート

## 目的

公開Notebook `phuongncn/kdrill-f594-ucont8` のU境界fadeだけを、現行ML submitted anchor exp287の固定OOFへ独立移植し、target-free freeze後に固定gateで監査できる状態へ実装する。

## 現在の状態

- Route: `ml_model`
- 状態: `kaggle_cpu_v2_complete_fail_close_no_rescue`
- 親: `exp287_fold_safe_formation_74_addonly_on_exp264`
- CV / LB: `8.135096925090597` / 未提出
- 実装: true
- 正規Notebook科学ロジック: true
- Kaggle package / push / run: true / true / true（version 2完了）
- inference / submission: false / false

## 2026-07-22 設計固定

### 根拠

- exp287は保存済みOOF CV `8.136708220359452`、Public LB `7.530`で現行ML submitted anchor。
- exp287のOOFは3,783,989 rows / 773 wells、SHA `8f026c5c5f6508fb142981832994c6ba9cded4940168c648a9df9f3e698c3913`。
- 公開Notebookは`gap_U`を最大8 ft打ち消し、240 MD-ftで指数減衰させる1 scoring cellを追加している。
- 公開Notebook全体にはsame-well contact等が混在するため、U式だけを切り出して手元OOFで検証する必要がある。

### 固定した変更

- 1 variant: `u_cap8_tau240_always_on`。
- `gap_U = parent_first_hidden + Z_first_hidden - (last_visible_TVT_input + Z_last_visible)`。
- `move = -clip(gap_U, -8, 8) * exp(-md_since_boundary / 240)`。
- 親、fold、raw境界、cap、tau、bucket、metrics、gateを固定。
- candidate／diagnosticをtruth-freeでSHA freezeし、その後だけtruthをjoinする。

### 実行予定量

- active postprocess variant: 1
- reporting folds: 5
- trained folds / model configs / trained models / LightGBM boosters: `0 / 0 / 0 / 0`
- PF / Beam / HMM well-runs: `0 / 0 / 0`
- parent/control retraining: 0
- GPU: 0
- inference / submission: 0 / 0

### 固定gate

- Technical: 親SHA・CV・row/well/ID parity、全well prefix/suffix契約、truth-before-freeze 0、finite、最大8 ft、単調fade、first-hidden gap非増加、SHA readbackをAND。
- Scientific: pooled gain`>=0.020 ft`、4/5 folds、0--240 gain`>=0.050 ft`、240--480 delta`<=+0.020`、480--1000`<=+0.010`、1000+`<=+0.005`、hidden-like各`<=+0.020`、by-well median`<=0`、p95`<=+0.10`、worst`<=+0.50 ft`をAND。
- FAIL時はcap/tau/threshold/gate/parent/distance範囲/blendの救済なしでcloseする。

## 再現性メモ

- `docs/06_reproducibility.md`確認済み。
- seed policy: `no_rng_fixed_saved_oof_postprocess`。
- stochastic components: なし。
- CPU/GPU: 将来のStage 0はKaggle private CPU、GPU 0、worker 1、internet off。
- deterministic anchor: false。補正式は固定入力に対してdeterministicだが、exp287 GPU parentのbitwise rerun parityは未確認。
- required SHA: public reference identity、parent OOF/model manifest、raw horizontal ordered manifest、schema、config、pretruth candidate/diagnostic、metrics、package/kernel。
- model SHA: 新規model 0。親manifest SHAを記録する。
- prediction / submission SHA: 現時点では生成物なし。将来のStage 0 prediction SHAは必須、submissionはinference承認後のみ。

## コマンドログ

### 実行済み

```bash
make new-steering EXP=exp349_exp287_u_boundary_continuity_fade
make new-exp EXP=exp349_exp287_u_boundary_continuity_fade
```

`task` CLIは環境に存在しなかったため、リポジトリが定義する同等のMakefile入口を使用した。

### 初期scaffold作成時点で未実行

- 実装、Jupytext変換、専用test、正規Notebook採用。
- Kaggle package、push、run、output取得。
- raw-test inference、submit-check、submission。

## 2026-07-22 実装

- ユーザーの`exp349を実装してください`を、事前登録済み固定仕様のimplementationとcompact self-contained Notebook候補の承認として記録した。
- `exp349_exp287_u_boundary_continuity_fade_compact_selfcontained_train.py/.ipynb`へ次を実装した。
  - exp287 OOF SHAを検証し、Stage Aでは`id/well/parent prediction`だけを列選択するpretruth projection。
  - raw horizontalを`MD/Z/TVT_input`だけで読み、全773 wellsにfinite prefix＋contiguous NaN suffix、strict MD、suffix ID parityを要求するfail-closed preflight。
  - `gap_U`、固定`cap=8/tau=240`のmove、fixed distance/gap/sign bucket、candidate、well診断。
  - candidate parquet、diagnostic、raw/schema/input manifestのSHA freezeとreadback検証。
  - freeze manifest SHA再検証後だけ`actual_tvt/outer_fold/hidden-like assignment`を開くStage B late-truth boundary。
  - pooled、5 folds、distance、gap、hidden-like、by-well metricsと固定technical/scientific AND gate。
  - `PASS_FOR_INFERENCE_REVIEW`または`FAIL_CLOSE_NO_RESCUE`の固定decisionと18生成物/reproducibility manifest契約。
- `exp349_exp287_u_boundary_continuity_fade_compact_selfcontained_inference.py/.ipynb`は、Stage 0 PASSと別承認前のraw-test prediction/submissionを拒否するfail-closed候補として実装した。
- `experiments/exp349_exp287_u_boundary_continuity_fade/tests/test_exp349_u_boundary_continuity_fade.py`へsynthetic tests 10件を追加した。
  - 1 postprocess / 0-training契約、inference fail-close、SHA同一copy resolver。
  - prefix/suffix、固定式、cap/fade、bucket、noncontiguous suffix・非単調MD拒否。
  - pretruth禁止列、raw/OOF ID mismatch、全scientific gate、far regression、freeze改ざん検知。
- 既存canonical train/inference Notebookは上書きせずscaffoldのまま保持した。
- configは`implementation_complete_no_run`、implementation approval trueへ更新した。canonical採用、Kaggle package/push/run、inference、submissionはfalseのまま。
- 親exp287にcompact self-contained train sourceは存在しない。親の正規train sourceは362行・6章、本候補は1,576行・9章で、Stage A/B、freeze、metrics、manifest、decisionをNotebook上へ展開している。
- model / LightGBM config / trained fold / booster / PF / Beam / HMM / control再学習 / GPUは`0 / 0 / 0 / 0 / 0 / 0 / 0 / 0 / 0`。active postprocess variant 1、reporting folds 5。

### 実装確認

```bash
.venv/bin/python -m py_compile experiments/exp349_exp287_u_boundary_continuity_fade/*compact_selfcontained*.py experiments/exp349_exp287_u_boundary_continuity_fade/tests/test_exp349_u_boundary_continuity_fade.py
.venv/bin/ruff check experiments/exp349_exp287_u_boundary_continuity_fade/*compact_selfcontained*.py experiments/exp349_exp287_u_boundary_continuity_fade/tests/test_exp349_u_boundary_continuity_fade.py
.venv/bin/pytest -q experiments/exp349_exp287_u_boundary_continuity_fade/tests/test_exp349_u_boundary_continuity_fade.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp349_exp287_u_boundary_continuity_fade/exp349_exp287_u_boundary_continuity_fade_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp349_exp287_u_boundary_continuity_fade/exp349_exp287_u_boundary_continuity_fade_compact_selfcontained_inference.py
make validate-exp EXP=exp349_exp287_u_boundary_continuity_fade
```

- synthetic tests: `10 passed`。
- Jupytext train/inference round-trip、py_compile、Ruff full、strict experiment validation: PASS。
- `review_exp_docs.py exp349 --root .`: core evidence categories present。
- implementation SHA:
  - compact train py / ipynb: `ef002148...319c6` / `2ab9b02f...21a5`
  - compact inference py / ipynb: `70e483fc...4df` / `54753d65...3e0`
  - dedicated tests: `2a1b1623...019a`
- repo全体`make test`: 650件中`643 passed / 5 skipped / 2 failed`。2件はいずれも既存`experiments/exp296_exp223_self_gr_known_tvt_support_gate/tests/test_exp296_exp223_self_gr_known_tvt_support_gate.py`が、完了済みexp296 configの`completed_train_side_guard_failed_closed`と`run_variant=false`ではなく旧running状態を期待する不一致で、exp349専用testは全件PASSした。範囲外のexp296 code/config/testは変更していない。
- Kaggle package/push/run、output取得、raw-test inference、submissionは実行していない。

## 実装完了時点の次のアクション（実施済み）

正規train Notebook採用とKaggle private CPU Stage 0 1回は別承認待ち。Stage 0全gate PASS後もraw-test inferenceとsubmissionは自動実行しない。

## 2026-07-22 Stage 0 実行承認

- ユーザーの`実行してください`を、compact self-contained train候補の正規train Notebook採用と、Kaggle private CPU Stage 0を1回実行する明示承認として記録した。
- push前の実行量を再確認した。active postprocess variant `1`、reporting folds `5`、trained folds / LightGBM configs / trained models / boostersは`0 / 0 / 0 / 0`、合計booster `0`。
- PF / Beam / HMM well-runsは`0 / 0 / 0`、親exp287/control再学習なし、GPUなし、internet off、inferenceなし、submissionなし。
- 入力は固定exp287 parent kernel `kentookumura/exp287-foldsafe-form74-addonly-exp264-train`のOOF/model manifest、公式competition raw train、固定hidden-like assignmentだけを使う。
- Kaggle CLIで親kernelの`fold_safe_formation_oof_predictions.parquet`と`model_manifest.json`の存在を確認した。
- canonical train Notebook SHAは`ee883af257c736658fc6d8df002300ca5aacb95784b8d0dddca3cc6f19b00f16`。Kaggle bootstrap済みnotebook SHAは`1a7feac7bb12ac0b0c22457382274ebb866284a36875838db5fe905811effa44`、kernel metadata SHAは`faab16e86be3d2506fadcf3411383a0cf421a99ecf67af15644ba56c91eb509b`、実行config SHAは`711b12f7ca7ffc88a6746477329480ce96247316f91cb73550412890ebc37927`。
- package metadataはprivate / CPU / GPU off / internet off / competition source 1件 / parent kernel source 1件 / run-on-pushを確認した。bootstrapに固定hidden-like assignmentを含むことも確認した。
- Stage 0 PASSの場合もraw-test inferenceとsubmissionは別承認まで実行しない。FAILの場合は事前登録どおり救済探索せずcloseする。

### Kaggle version 1

- kernel: `kentookumura/exp349-exp287-u-boundary-continuity-fade-train` version 1。
- push後にremoteをpullし、21セルの結合sourceとprivate / CPU / GPU off / internet off / competition source / parent kernel sourceがlocal packageと一致することを確認した。
- 約102秒でtechnical error。親OOF/rawの読み込み後、Stage Aのgap bucket文字列化でKaggle pandasが`numpy.ndarray`を返したのに`.to_numpy()`を呼んだため停止した。
- target-free candidate freeze、truth late-join、科学評価には未到達。cap/tau/variant/gate/入力は変更しない。
- technical-only修正として、pandasの返り値型に依存しない`np.asarray(bucket.astype(str), dtype=str)`へ置換し、同一kernel IDでversion 2を再実行する。
- version 2のcompact train source / canonical train Notebook / Kaggle bootstrap notebook SHAは、それぞれ`e9182e0b121a6414c6b17a8fe2cb1d31e59320011f3b2a2da731f1b8a5951250` / `e564b836d6fc2fcf70c8348bb0cd66857ef54133f6dd9c45b6a412df335846f3` / `d6f540bbec35b703db4a5e3abe3ed258b9b34caf5f742f11bffb38342cdfbba7`。
- version 2再梱包後も専用test 10件、Jupytext round-trip、py_compile、Ruff、strict experiment validationはPASSした。

## 2026-07-23 Kaggle version 2 完了確認

- ユーザーから完了連絡を受け、`kaggle kernels logs kentookumura/exp349-exp287-u-boundary-continuity-fade-train`でversion 2の正常完了を確認した。
- kernel id_noは`128239658`。core resultは約`156.134 sec`、notebook変換まで含む最終ログ時刻は約`165.895 sec`。
- 入力／実行量は事前契約どおり、1 postprocess variant / 5 reporting folds / trained fold・model config・trained model・LightGBM booster・PF・Beam・HMM・親control再学習・GPUすべて0。hidden-test predictionとsubmissionは0。
- Stage Aは親OOF 3,783,989 rows / 773 wells、SHA`8f026c5c...c3913`とraw suffixを完全整列した。truth、outer fold、hidden-like assignmentを開く前にcandidate/diagnosticをfreezeし、truth access count 0を確認した。
- Stage A freeze manifest SHAは`a1070b4a7f9bcaaa1d973bd10719ee2fe9daf29dce74899e0c67e5f0f2a64d99`、candidate SHAは`adf51375d0d9b676dd9b7610528da43a862801632eed3afd61f340b3639fcbce`、diagnostic SHAは`b6be5278cafed4cfa5ed51ca754bd58321ac1ffbe0007e5aa847977468e19ef0`。
- Technical gateは全PASS。最大補正絶対値`6.282790530 ft`、formula差`0`、finite、単調fade、first-hidden gap非増加、row/well/ID/CV/SHA parityを確認した。

### 科学結果

- pooled: parent `8.136708220359452` → candidate `8.135096925090597`、改善`0.001611295268855173 ft`。
- fold改善: fold 0--4で`0.001007899 / 0.001660956 / 0.001835165 / 0.001488494 / 0.002055754 ft`、5/5改善。
- distance改善: 0--64 / 64--128 / 128--240 / 240--480 / 480--1000 / 1000+で`0.305682 / 0.135888 / 0.068317 / 0.027269 / 0.002901 / 0.000002 ft`。primary 0--240は`0.110003778 ft`改善。
- hidden-like spatial / typewell-purgedは`0.002098746 / 0.002239703 ft`改善。
- by-well delta median / p95 / worstは`-0.000363396 / +0.010450856 / +0.063651341 ft`。`|delta|>=0.25 ft`の大幅改善／悪化wellはともに0。
- Scientific gateは11/12 PASS。唯一のFAILはpooled改善`0.001611295 ft < 0.020 ft`。
- 最終判定は`FAIL_CLOSE_NO_RESCUE`。境界近傍は明確に改善したが、3,012,442 rowsを占める1000+が実質無変化で、anchor更新に必要な総改善量に届かなかった。

### 証跡と終端判断

- Kaggle output archive全体は取得せず、fold／distance／hidden-like／by-well／decision／reproducibility等の小規模metrics/manifestsだけを`--file-pattern`で選択取得した。
- 取得した小規模生成物はreproducibility manifest記録SHAと全件一致した。Kaggle metrics SHAは`9217bac22f5cb053a1e343402dffe8af1cb6e1eb20e4ba2891737f65e3a920b8`、decision SHAは`cbd821a4ae57bcd8cd0d5516d9661e306164f38ad48746ef8a83f8b88431362a`。
- `cap=8/tau=240`、gap threshold、distance範囲、well gate、blend、親、scientific gateを同じOOFで救済しない。direct fixed U-boundary fade branchを閉じる。
- raw-test inference、submission、Public LB probeは行わない。continuity再訪はexp349のpost-hoc救済ではなく、独立したtarget-free add-only feature／selector仮説を事前設計できた場合だけ別途判断する。
