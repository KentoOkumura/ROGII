# exp347_prefix_gr_unary_batched_window_exact_ssm セッションノート

## 目的

exp332の固定window exact structured objectiveを維持し、4-window batched DPだけで計算gateを通せるか検証する。

## 現在の状態

- Route: `ensemble`
- 状態: 固定16-window Stage 0 technical parity FAIL、terminal close
- 親: `exp332_prefix_gr_unary_fixed_window_structured_ssm`（Stage 0 runtime FAILでterminal close済み）
- CV / LB: なし / なし
- Notebook: compact self-contained train候補を正規trainへ採用。正規inferenceはscaffold placeholderを維持
- compact source / tests: あり / あり
- Kaggle package / Stage 0 output: あり / あり。Stage A model / prediction / submission: なし / なし / なし

## 2026-07-22 設計確定

- ユーザー依頼により、先に提示した案1「複数windowをまとめてGPU計算する」をexp347として採番した。
- exp332は再開せず、結果・config・Notebookを変更しない。exp347はそのcompute-faithful follow-upとして独立管理する。
- 唯一の変更は、exp332の`batch 1 × gradient accumulation 4`を`batch 4 × accumulation 1`へ置き換えること。連続するfrozen schedule 4件のper-window normalized loss平均を1 optimizer updateへ渡す。
- row/position/rate paddingは`-inf` potentialと明示maskで除外し、dropout offの固定4 windowsでscalar loss/posterior/gradient/1-step update parityを先に要求する。
- 256 rows、3 slots、最大3 windows/well/epoch、8 epochs、structured NLL`1.0`、sigma`0.35 ft`、local CE`0.25`、architecture、teacher boundary、41-rate exp209 grammar、full-well controls/gateはexp332から固定する。
- Stage 0は1 benchmark variant / 固定16 windows / temporary neural model 1。persisted model、trained fold、LightGBM config、booster、PF/Beam、parent/control再学習は`0/0/0/0/0/0`。
- compute gateはT4保守的fold外挿`<=8.5 h`、peak`<=14 GB`、exp332 `13.151137 h`比speedup`>=1.55x`。FAIL後のbatch size、padding、compile/fused kernel、science contract救済は禁止する。
- Stage Aは全Stage 0 gate PASSと別承認後だけfold 0 / architecture 1 / seed 42 / neural model 1。Stage B/C、推論、提出も別承認とする。
- exp348 path-rankingより先行するP2とする。ただし現行P1実験を追い越さない。

## 2026-07-22 実装

- ユーザー依頼「exp347を実装してください」を、compact self-contained train/inference候補、batched exact DP、専用contract tests、設定・記録の実装承認として扱った。正規Notebook上書き、Kaggle package/push/run、Stage A、推論、提出は承認範囲外である。
- exp332 compact self-contained trainを構成参照にし、科学契約を維持したまま`BatchedStateSpec`を追加した。active windowごとのrow/position/rate maskとinactive dummyを作り、padding potentialを`-1e18`へ固定してposterior/loss/gradientから除外する。
- exact DPはrow scanだけを逐次に残し、initial prior、rate transition、position transition、normal/label-conditioned forward-backwardをbatch次元でvectorizeした。出力posteriorはinvalid row/positionを明示的に0へ戻す。
- trainingはfrozen scheduleの連続4 windowsを再ソートせずchunk化する。各windowのstructured NLLとlocal CEをvalid rowで正規化し、active windowの算術平均を1回backwardしてAdamWを1 step進める。最終不足batchはinactive dummyでpadする。
- Stage 0は固定16 windowsの先頭4件でscalar/batch loss、partition、posterior、unary gradient、AdamW 1-step、padding exclusion、finiteをreportする。その後4 batchesでtrain/forward-onlyを計測し、full-well controlはsuffix長stable sort後の4-well batched decodeで測る。
- Stage A候補も同じ4-window trainingへ移し、outer-validはtruthを読まずsuffix長stable sortしてreal/shuffle/geometryを4-well batch decodeする。global Viterbi診断だけは各well scalar tracebackを維持する。
- scalar/batch parityでは追加neural modelを複製せず、同じtemporary unary model 1個からdropout-off unaryをfreezeし、unary tensorをparameterとしてgradient/AdamW updateを比較する。Stage 0 temporary neural model数は1のままである。
- fail-closed inference候補はbatched full-official-suffix contractだけを検証し、Stage B promotion前にはsubmissionを生成しない。
- 親compact trainは`3,045`行、exp347候補は`4,090`行。13章を維持し、batched state/padding、training、parity、Stage 0、freeze-first Stage AをNotebook上で追える。

### 実装検証

```text
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <compact_train.py> <compact_inference.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <compact_train.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <compact_inference.py>
.venv/bin/python -m py_compile <compact_train.py> <compact_inference.py> experiments/exp347_prefix_gr_unary_batched_window_exact_ssm/tests/test_exp347_prefix_gr_unary_batched_window_exact_ssm.py
.venv/bin/ruff check <compact_train.py> <compact_inference.py> experiments/exp347_prefix_gr_unary_batched_window_exact_ssm/tests/test_exp347_prefix_gr_unary_batched_window_exact_ssm.py --select E,F,I,UP,B
.venv/bin/pytest -q experiments/exp347_prefix_gr_unary_batched_window_exact_ssm/tests/test_exp347_prefix_gr_unary_batched_window_exact_ssm.py
make validate-exp EXP=exp347_prefix_gr_unary_batched_window_exact_ssm
```

- Jupytext変換/`--test`、py_compile、Ruff、strict experiment validation: PASS。
- 専用pytest: `16 passed, 2 skipped`。skip 2件はローカル環境にPyTorchがなく、scalar exact gradientとscalar/batched numerical parityを実行できないため。Kaggle T4 Stage 0で同じ数値契約を必須gateとして実行する。
- 全体pytest: `643 passed, 5 skipped, 2 failed`。2件はいずれも既存`experiments/exp296_exp223_self_gr_known_tvt_support_gate/tests/test_exp296_exp223_self_gr_known_tvt_support_gate.py`が、完了済みexp296の現status/run flagに対して旧Kaggle実行前状態を期待する既知不整合で、exp347専用testは全件PASSした。
- `__file__`参照0、canonical Notebook上書き0、Kaggle package/push/run 0。

## 実行量ガード

- 2026-07-22のユーザー依頼「実行してください」を、固定16-window Kaggle T4 Stage 0だけの実行承認として扱う。Stage A/B/C、推論、提出は承認範囲外である。
- 今回実行: active benchmark variant / fixed windows / temporary neural model / persisted model / trained fold / LightGBM config / booster / PF-Beam / control再学習 / parent再学習 = `1 / 16 / 1 / 0 / 0 / 0 / 0 / 0 / 0 / 0`。
- compact self-contained train候補をcanonical train Notebookへ採用する。canonical inferenceは変更しない。
- Kaggle GPU shapeは`NvidiaTeslaT4`、internet off、kernel sourceは保存済みexp209/exp115を参照する。baseline/controlは再学習しない。
- 将来Stage A承認時: active architecture 1 / fold 0 / seed 42 / neural model 1。control再学習0。

## コマンドログ

```text
make new-steering EXP=exp347_prefix_gr_unary_batched_window_exact_ssm
make new-exp EXP=exp347_prefix_gr_unary_batched_window_exact_ssm
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp347_prefix_gr_unary_batched_window_exact_ssm/exp347_prefix_gr_unary_batched_window_exact_ssm_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp347_prefix_gr_unary_batched_window_exact_ssm/exp347_prefix_gr_unary_batched_window_exact_ssm_compact_selfcontained_inference.py
make validate-exp EXP=exp347_prefix_gr_unary_batched_window_exact_ssm
```

## 2026-07-22 Stage 0 pre-push

- canonical kernel id/titleを`kentookumura/exp347-prefix-gr-batched-window-ssm-stage0` / `exp347 prefix gr batched window ssm stage0`へ固定した。slugは42文字でtitle由来slugと一致する。
- package metadataはprivate、T4、internet off、competition source 1、保存済みexp209/exp115 kernel source 2、run-on-pushを確認した。
- package Notebook / metadata / config SHA256は`1c3e2d554c4b6964d3075bf13aaac6c28478dd3b48bac2d2236ee9968519aa4c` / `1e85b94b91d06f1a3d390a4dae6e71405c2467bbf2a287249878ea448a7cf098` / `376c03da9b6e122cd9fe32c95f3edf079fca5e7126e13aa72700307809bbb51f`。
- credential checkはOAuth/legacy CLI credentialが有効。canonical idのpre-push pullは403で、既存kernelがないことを確認した。
- canonical採用後のJupytext `--test`、py_compile、Ruff、専用pytest `16 passed, 2 skipped`、strict experiment validation、template validationはPASS。local torch未導入によるnumerical test 2件はKaggle T4 gateで実行する。
- canonical kernel version 1のpushに成功した。Stage 0実行中であり、重複pushせず同versionを監視する。
- Kaggle側pull-backで`id_no=128239400`、private、T4、internet off、competition/exp209/exp115 source一致を確認した。
- ユーザー指示「監視は止めていいです。完了したら連絡します。」により、Kaggle kernel自体は停止せずversion 1のローカル監視だけを終了した。最終確認時statusは`RUNNING`、logsは空。ユーザー連絡後にstatus/logs/outputを取得してStage 0 gateを判定する。

```text
.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp347_prefix_gr_unary_batched_window_exact_ssm --notebook train --kernel-id kentookumura/exp347-prefix-gr-batched-window-ssm-stage0 --title "exp347 prefix gr batched window ssm stage0" --run-on-push --strict
kaggle kernels pull kentookumura/exp347-prefix-gr-batched-window-ssm-stage0 -p /tmp/exp347-stage0-prepush -m
kaggle kernels push -p experiments/exp347_prefix_gr_unary_batched_window_exact_ssm/kaggle/train --accelerator NvidiaTeslaT4
```

## 2026-07-23 Stage 0完了

- ユーザーの完了連絡後、canonical kernel version 1（id_no `128239400`）が`COMPLETE`であることと完了ログを確認した。report生成時刻は`2026-07-22T14:13:03.524337+00:00`、Notebookログ上のreport出力は開始後約`671.323 sec`だった。
- 実行量は契約どおりactive benchmark variant 1 / fixed windows 16 / temporary neural model 1 / persisted model 0 / trained fold 0 / LightGBM config 0 / booster 0 / PF-Beam 0 / control・parent再学習0。outer-valid truth access 0、Stage A model 0。
- scalar/batch parityはloss `0.0`、partition `0.0`、gradient `1.4319085e-8`、AdamW 1-step `0.0`、invalid posterior/gradient `0.0/0.0`、finite rate `1.0`でPASSしたが、posterior max abs error `1.4662743e-5`が固定上限`1e-6`をFAILした。
- 計算gateはp50/保守的fold外挿`4.741982 / 5.108737 h`、exp332比speedup`2.574244x`、peak memory`5.928168 GB`で、`<=8.5 h` / `>=1.55x` / `<=14 GB`をすべてPASSした。
- reportの`finite_pass=false`は実データの非有限を意味しない。parity finite checkと全52 measurementの正時間はPASSしているが、実装が`finite_pass = technical_pass and measurement_finite`と定義しているためposterior parity FAILに連動した。
- outputを`kaggle/output/stage0_v1`へ取得した。measurementは52行（structured train 4、forward-only 4、full-well unary 32、batched exact decode 12）、window/boundary manifest各16行、batch padding manifest 68行。
- selection / boundary / padding / scalar parity / measurement / report / log SHA256はそれぞれ`b78ed92d...1e89` / `664b3fc7...1d1` / `28f30e4e...49b1` / `3822eddc...51e` / `5c3f89eb...e4a8` / `e8a706ba...e454` / `53310887...ed6`で、report内SHAと実ファイルが一致した。
- AND gateは`technical_parity_pass=false`、総合`passed=false`、decision `close_without_batch_or_science_rescue`。事前契約どおりStage A/B/C、推論、提出、batch size、padding、compile/fused kernel、閾値、科学契約の救済を行わずbranchを閉じる。

```text
kaggle kernels status kentookumura/exp347-prefix-gr-batched-window-ssm-stage0
kaggle kernels logs kentookumura/exp347-prefix-gr-batched-window-ssm-stage0
kaggle kernels output kentookumura/exp347-prefix-gr-batched-window-ssm-stage0 -p experiments/exp347_prefix_gr_unary_batched_window_exact_ssm/kaggle/output/stage0_v1
```

## 再現性メモ

- seed policy: seed 42 + stable SHA256 window/batch/boundary/decode order。
- stochastic components: CUDA convolution、AMP、AdamW、dropout、dataloader order。
- deterministic anchor: false。
- 記録済み: window/batch/boundary/padding、scalar parity、measurement、report、package/kernel/log SHA。Stage A model、prediction、submissionは未生成。gzip生成物なし。

## 次のアクション

1. exp347はterminal closeとして維持し、再pushやStage A/B/Cへ進まない。
2. 同系のbatch/padding/compile/fused kernel/閾値/科学契約救済を追加しない。
3. 独立仮説のexp348は先行条件を満たしたが、高リスクP3の別実験としてユーザーの別判断を待つ。
