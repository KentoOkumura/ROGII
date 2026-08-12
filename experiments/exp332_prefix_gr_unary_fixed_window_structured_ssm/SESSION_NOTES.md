# exp332_prefix_gr_unary_fixed_window_structured_ssm セッションノート

## 目的

exp295のsoft structured objectiveを固定長windowでcompute-feasibleにする代替設計を、local CE案と分離して固定する。

## 現在の状態

- Route: `ensemble`
- 状態: Stage 0 runtime gate FAIL / branch closed
- CV / LB: なし / なし
- implementation / Stage 0 completed / Stage A / inference / submission: あり / FAIL完了 / 閉鎖 / 閉鎖 / 閉鎖
- Stage 0 benchmark variant / temporary neural model / persisted Stage A model / LightGBM config / booster / PF-Beam run: `1 / 1 / 0 / 0 / 0 / 0`

## 2026-07-21 設計確定

- ユーザー依頼によりfixed-window structured training案をexp332として採番した。
- windowは256 rows、3 scheduled slots/well/epoch、最大3 active non-overlap windows。最大556 fit wellsなら1,668 active windows / 427,008 scored positions/epochを上限に固定した。
- objectiveはGaussian soft-label structured NLL`1.0`、sigma`0.35 ft`、local CE`0.25`で、exp295 version 3と同じ。
- interior teacher boundaryはloss初期化だけに限定し、encoder/valid/test inputから除外した。
- full-well evaluation、controls、Stage A/B/C promotion gateはexp331/exp295と揃えた。
- exp331を推奨先行案とし、同時実装・同時GPU比較を禁止した。
- 実装、Notebook編集、Kaggle実行、推論、提出は行っていない。

## 2026-07-22 実装

- exp331がStage Aで`real_rmse_vs_exp209`、well p95、worst-wellの科学gateをFAILし、branch close済みであることを確認した。
- ユーザー依頼「exp332を実装してください」を、compact self-contained Stage 0/Stage A候補、fail-closed inference候補、専用test、設定・記録の実装承認として扱った。Kaggle package/push、Stage 0実行、Stage A学習、推論、提出は承認範囲外である。
- truth/errorを読まない`select_window_slots`で全8 epochs、3 scheduled slots/well/epochをfreezeする。slot 0はofficial suffix先頭、slot 1/2は後続slotの収容余地を残すstable SHA256順位で選び、256-row full windowを置けないslotだけinactiveにする。
- schedule manifest保存後にteacher boundary manifestを生成する。interior windowは直前truth TVTと直前2 rowのrateを使うが、rateは固定41-state gridへ量子化し、window StateSpecの初期priorだけへ渡す。encoderは常にofficial `TVT_input`のままである。
- exp295のGaussian soft-label structured NLL`1.0`、sigma`0.35 ft`、local CE`0.25`を復元した。通常/label-conditioned forward-backwardの4 sweepsを256-row window内だけで行う。
- Stage 0はsuffix長quartileごと4件、計16 windowsを固定し、structured forward/backward/optimizer、同objective forward-only、freeze後full-well real/shuffle/geometry decodeを計測して8.5時間/14 GBへ外挿する。
- Stage Aはschedule/boundary/input SHAを保存し、1 architecture × fold 0 × seed 42 = 1 neural modelだけを学習する実装とした。LightGBM config/booster、PF/Beam、parent/control再学習は`0/0/0/0`。outer-validはofficial prefixからfull-well exact SSMでdecodeし、truthはprediction freeze後に読む。
- `execution.selected_stage=implementation_only`、push/GPU/inference/submission flagはfalse。正規Notebook scaffoldは上書きせず、別名compact self-contained候補を生成した。
- 親compact trainはexp295 `2,099`行、直近実装参照exp331 `2,575`行、exp332 `3,045`行。exp332は13章を維持し、window schedule、teacher boundary、structured objective、Stage 0、full-well Stage AをNotebook上で追える。

### 実装検証

```text
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <compact_train.py> <compact_inference.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <compact_train.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <compact_inference.py>
.venv/bin/python -m py_compile <compact_train.py> <compact_inference.py> experiments/exp332_prefix_gr_unary_fixed_window_structured_ssm/tests/test_exp332_prefix_gr_unary_fixed_window_structured_ssm.py
.venv/bin/ruff check <compact_train.py> <compact_inference.py> experiments/exp332_prefix_gr_unary_fixed_window_structured_ssm/tests/test_exp332_prefix_gr_unary_fixed_window_structured_ssm.py --select E,F,I,UP,B
.venv/bin/pytest -q experiments/exp332_prefix_gr_unary_fixed_window_structured_ssm/tests/test_exp332_prefix_gr_unary_fixed_window_structured_ssm.py
```

- Jupytext `--test`、py_compile、Ruff: PASS。
- strict experiment validation、template validation: PASS。
- 専用pytest: `14 passed, 1 skipped`。skipはローカル環境にPyTorchがない場合のexact structured gradient test。
- 全体pytest: `548 passed, 3 skipped, 2 failed`。失敗は既存`experiments/exp296_exp223_self_gr_known_tvt_support_gate/tests/test_exp296_exp223_self_gr_known_tvt_support_gate.py`の2件で、完了済みexp296の現status/run flagに対して旧実行前状態を期待している。今回のexp332変更由来ではない。
- `__file__`参照0、canonical Notebook上書き0、Kaggle package/push/run 0。

## 2026-07-22 Stage 0実行承認

- ユーザー依頼「実行してください」を、固定16-window T4 Stage 0 microbenchmarkのcanonical train採用、package、Kaggle push、完了確認、gate判定、report SHA確認の承認として扱った。
- 実行variantはStage 0 benchmark 1件。同一の一時的なbenchmark neural model 1個を16 windowsで逐次更新し、model freeze後にreal / circular-shuffle / geometry-onlyをdecodeする。control用の別モデルは学習しない。
- trained fold 0、永続Stage A model 0、LightGBM config 0、booster 0、PF/Beam run 0、parent/control再学習0。Stage A予定の`1 architecture × fold 0 × seed 42 = 1 neural model`は今回実行しない。
- Stage 0がPASSしてもreport SHAを固定した上でStage A fold 0は別承認を必要とする。Stage B/C、推論、提出も未承認のまま維持する。
- compact self-contained train候補をcanonical train Notebookへ採用する。GPU shapeは`NvidiaTeslaT4`に固定する。
- canonical trainとcompact候補のSHAはともに`3b921d5a...f1ef80`で一致した。package Notebook SHAは`1f1d506f...ae82f`、metadata SHAは`366bfb05...04ef`、同梱config SHAは`9640fa9f...a0f6`。bootstrap manifest内のconfig SHA、`stage0_microbenchmark`、push承認、Stage A未承認、1 benchmark variant / 1 temporary model / 0 persisted model / 0 booster、T4、internet offを確認した。
- push前のcanonical kernel pullは403、既知の自分のexp331 private kernel pullは成功したため、認証障害ではなくexp332 Stage 0 kernelが未作成であることを確認した。
- 初回の57文字kernel id/title `kentookumura/exp332-prefix-gr-unary-fixed-window-structured-ssm-stage0` / `exp332 prefix gr unary fixed window structured ssm stage0`は`SaveKernel 400`となり、直後のpullも403でkernel未作成だった。id/title slugは一致していたため、Kaggleの長さ制約に当たったと推定する。
- 同じexpのまま科学設定を変えず、46文字のcanonical id/title `kentookumura/exp332-prefix-gr-unary-fixed-window-ssm-stage0` / `exp332 prefix gr unary fixed window ssm stage0`へ短縮した。再packageのNotebook SHAは`1f1d506f...ae82f`で不変、metadata SHAは`6c7d13ff...2434`、同梱config SHAは`9640fa9f...a0f6`。
- 短縮後のcanonical kernel version 1をpushし、Kaggle側pull-backに成功した。`id_no=128231704`、private、`enable_gpu=true`、`machine_shape=NvidiaTeslaT4`、internet off、competition/exp209/exp115 source一致を確認した。
- `2026-07-22 12:31 UTC`までstatusは`RUNNING`。ユーザー指示「監視は止めていいです。完了したら連絡します。」に従い、ローカルの45秒監視ループだけを停止した。Kaggle kernel version 1は停止・再pushせず、そのまま継続する。完了連絡後にlogs/output、Stage 0 report、measurement、gate、SHAを確認する。

## 2026-07-22 Stage 0実行結果

- Kaggle kernel `kentookumura/exp332-prefix-gr-unary-fixed-window-ssm-stage0` version 1（id_no `128231704`）は`COMPLETE`。report生成時刻は`2026-07-22T12:47:58.614053+00:00`、Notebookログ上のreport出力は開始後約`1325.595 sec`だった。
- 固定16 windowsはsuffix-length quartileごと4件、16 wells。measurementは112行で、structured forward/backward/optimizer 16、同objective forward-only 16、full-well unary real/shuffle 32、frozen exact SSM decode 48（real / circular-shuffle / geometry-only各16）。全`seconds`は正だった。
- p50 fold外挿は`12.744535682 h`、固定p10-throughputによる保守的外挿は`13.151137275 h`で、runtime gate `<=8.5 h`をFAILした。peak GPU memoryは`1.203262806 GB`でmemory gate `<=14 GB`をPASSした。
- 保守的外挿の内訳はfit structured training `9.214264 h`（70.06%）、early-stop structured objective `0.998164 h`（7.59%）、3-control full-well decode `2.937457 h`（22.34%）、real/shuffle unary `0.001253 h`。固定window化しても4-sweep exact DPとfull-well decodeがKaggle時間内に収まらない。
- `outer_valid_truth_access_count=0`、`trained_stage_a_model_count=0`。Stage 0 boundary manifestは16件すべてofficial-prefix slot 0で、encoder sourceは全件`official_prefix_only`。Stage A学習、parent/control再学習、LightGBM、PF/Beam、推論、提出は0件。
- selection / teacher-boundary / measurement SHAは順に`03c1eeb0...f15c` / `1ace760b...3159` / `ae67b0ad...d8d1`で、report内記録と実ファイルSHAが一致した。report SHAは`acdadad6...ba8e`、Kaggle log SHAは`97ecddbe...7378`。
- 事前契約どおりdecisionは`close_without_window_or_loss_rescue`。window長/数、boundary、loss weight/sigma、architecture、decoder、epoch/viewの救済を行わずbranchを閉じる。実行承認は消費済みとして`selected_stage=implementation_only`、`kaggle_push_approved=false`へ戻した。Stage A/B/C、推論、提出は開始しない。
- exp331 local-CEは計算gateを通ったが科学gateをFAILし、exp332 structured-windowは計算gateをFAILしたため、exp295 neural unary familyの同系救済backlogは追加しない。既存の独立route候補の優先順位を維持する。

## コマンドログ

```text
make new-steering EXP=exp332_prefix_gr_unary_fixed_window_structured_ssm
make new-exp EXP=exp332_prefix_gr_unary_fixed_window_structured_ssm
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp332_prefix_gr_unary_fixed_window_structured_ssm/exp332_prefix_gr_unary_fixed_window_structured_ssm_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp332_prefix_gr_unary_fixed_window_structured_ssm/exp332_prefix_gr_unary_fixed_window_structured_ssm_compact_selfcontained_inference.py
make update-summary
make validate-exp EXP=exp332_prefix_gr_unary_fixed_window_structured_ssm
make validate-template
make test
.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp332_prefix_gr_unary_fixed_window_structured_ssm --notebook train --kernel-id kentookumura/exp332-prefix-gr-unary-fixed-window-ssm-stage0 --title "exp332 prefix gr unary fixed window ssm stage0" --run-on-push --strict
kaggle kernels push -p experiments/exp332_prefix_gr_unary_fixed_window_structured_ssm/kaggle/train --accelerator NvidiaTeslaT4
kaggle kernels pull kentookumura/exp332-prefix-gr-unary-fixed-window-ssm-stage0 -p /tmp/exp332-stage0-identity-v1 -m
kaggle kernels logs kentookumura/exp332-prefix-gr-unary-fixed-window-ssm-stage0
kaggle kernels output kentookumura/exp332-prefix-gr-unary-fixed-window-ssm-stage0 -p experiments/exp332_prefix_gr_unary_fixed_window_structured_ssm/kaggle/output/stage0_v1
```

## 再現性メモ

- seed policy: seed 42 + stable SHA256 window manifest
- stochastic components: CUDA convolution、AdamW、dropout（実装時）
- deterministic anchor: false
- SHA: input/fold/window/boundary/model/window posterior/full posterior/prediction/package/kernelを記録する設計

## 次のアクション

1. branch close。Stage A/B/C、推論、提出へ進まない。
2. exp332内のwindow/loss/decoder/runtime救済を行わない。
3. KAGGLE_DIRECTIONの既存独立候補を維持する。
