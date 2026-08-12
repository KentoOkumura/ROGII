# exp331_prefix_gr_unary_local_ce_exact_ssm セッションノート

## 目的

exp295のcomplete-well GR unaryをlocal CEだけで学習し、fixed exact SSMを評価時だけ使うcompute-feasibleな後継仮説を設計する。

## 現在の状態

- Route: `ensemble`
- 状態: Stage 0 PASS / Stage A科学gate FAIL / branch closed
- fold 0 CV / LB: `24.760360` / なし
- implementation / Stage 0 completed / Stage A completed / Stage B / inference / submission: あり / あり / FAIL完了 / 閉鎖 / 閉鎖 / 閉鎖
- Stage 0 benchmark variant / temporary benchmark neural model / trained fold / persisted Stage A model: `1 / 1 / 1 / 1`
- LightGBM config / booster / PF-Beam run / parent-control retraining: `0 / 0 / 0 / 0`

## 2026-07-21 設計確定

- ユーザー依頼により、exp295後継2案のうち推奨案をexp331として採番した。
- local hard CE`1.0`のみで全suffix rowを学習し、structured NLL/SSM callをtrainingとearly stoppingから0に固定した。
- exact SSMはmodel freeze後のfold評価、controls、承認後inferenceだけで使う。
- 固定16-view T4 microbenchmark、8.5時間、14 GB gateをfull Stage A前に置いた。
- Stage A規模は`1 architecture × fold 0 × seed 42 = 1 neural model`。LightGBM config/booster、PF/Beam、parent/control再学習はすべて0。
- Stage Bは別承認後のfold 1--4追加4 models、Stage Cはpromotion PASS後の別承認とした。
- この時点では実装、Notebook編集、Kaggle実行、推論、提出は行っていない。

## 2026-07-21 実装

- ユーザー依頼「exp331を実装してください」を、compact self-contained Stage 0/Stage A候補、fail-closed inference候補、専用test、設定・記録の実装承認として扱った。
- exp295 compact self-contained trainのinput/fold/preprocessing/encoder/fixed exp209 decoder/truth-late readoutを維持し、training lossとouter-train early stoppingをhard nearest-state local CE`1.0`だけに置換した。
- structured loss class、Gaussian label emission、optimizer/early stopping内のexact forward-backward呼び出しを削除した。exact SSMは固定16-view benchmarkのmodel-freeze後decodeとStage A model-freeze後評価にだけ残した。
- Stage 0はfold 0 outer-train viewをsuffix長quartileへ分け、各4件をstable SHA256順に選ぶ固定16 viewsとした。local-CE forward/backward/optimizer、forward-only、real/shuffle/geometry-only exact decodeを別計測し、p50と固定p10-throughput由来の保守的fold外挿を保存する。training/forward throughputはCSV読み込み・preprocessingを含むend-to-end時間で計算し、GPU-only時間も併記する。
- Stage A readoutは`md_since>=1000`とexp115のhidden-like spatial/typewell-purged rolesをprediction freeze後に読み、real/shuffle/geometry/exp209のsubgroup RMSEとassignment SHAを保存する。
- `execution.selected_stage=implementation_only`、`kaggle_push_approved=false`、`stage_a_gpu_approved=false`でfail-closedにした。Stage AはStage 0 PASSとreport SHA、別承認がなければ開始できない。
- Stage A実行予定規模は`1 architecture × fold 0 × seed 42 = 1 neural model`。LightGBM config 0、booster 0、PF/Beam run 0、control/parent再学習0である。
- 既存の正規Notebook scaffoldは上書きせず、別名compact self-contained `.py` / `.ipynb`を作成した。Kaggle package/push/run、Stage A学習、inference、submissionは0件。
- 親compact trainは2,099行、exp331 compact trainは2,558行。後者はStage 0選定・計測・外挿とlong-tail/hidden-like readoutを追加し、13章すべてをNotebookセルで追える。

### 実装検証

```text
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <compact_train.py> <compact_inference.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <compact_train.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <compact_inference.py>
.venv/bin/python -m py_compile <compact_train.py> <compact_inference.py>
.venv/bin/ruff check <compact_train.py> <compact_inference.py> experiments/exp331_prefix_gr_unary_local_ce_exact_ssm/tests/test_exp331_prefix_gr_unary_local_ce_exact_ssm.py --select E,F,I,UP,B
.venv/bin/pytest -q experiments/exp331_prefix_gr_unary_local_ce_exact_ssm/tests/test_exp331_prefix_gr_unary_local_ce_exact_ssm.py
```

- Jupytext `--test`、py_compile: PASS。
- 専用pytest: `16 passed, 1 skipped`。skipはローカル環境にPyTorchがない場合のexact posterior test。
- repository全pytest: `482 passed, 2 skipped, 2 failed`。2件は未変更のexp296で、実行完了後config status/run flagと旧test期待値が不一致な既存状態。exp331専用testとは独立のためexp296は変更していない。
- Ruff（E/F/I/UP/B）、strict experiment validation、template validation: PASS。
- `__file__`参照: 0。canonical Notebook上書き: 0。Kaggle package/push/run: 0。

## 2026-07-21 Stage 0実行承認

- ユーザー依頼「実行してください」を、固定16-view Stage 0 T4 microbenchmarkのcanonical train採用、package、Kaggle push、完了確認、成果物取得の承認として扱った。
- 実行variantはStage 0 benchmark 1件。同一の一時的なbenchmark neural model 1個を16 viewsで逐次更新し、model freeze後にreal / circular-shuffle / geometry-onlyをdecodeする。control用の別モデルは学習しない。
- trained fold 0、永続化するStage A model 0、LightGBM config 0、booster 0、PF/Beam run 0、parent/control再学習0。Stage A予定の`1 architecture × fold 0 × seed 42 = 1 neural model`は今回実行しない。
- Stage 0がPASSしても、report SHAを固定した上でStage A fold 0は別承認を必要とする。inference / submissionも未承認のまま維持する。
- compact self-contained train候補をcanonical train Notebookへ採用し、kernel id/titleは`kentookumura/exp331-prefix-gr-unary-local-ce-exact-ssm-stage0` / `exp331 prefix gr unary local ce exact ssm stage0`、GPU shapeは`NvidiaTeslaT4`に固定する。

## 2026-07-21 Stage 0実行結果

- Kaggle kernel `kentookumura/exp331-prefix-gr-unary-local-ce-exact-ssm-stage0` version 1は`COMPLETE`。report生成時刻は`2026-07-21T10:44:34.119829+00:00`、Notebookログ上のreport出力は開始後約1,207秒だった。
- 固定16 viewsはsuffix-length quartileごとに4件ずつ。measurement 80行はlocal-CE forward/backward/optimizer 16、forward-only 16、frozen exact SSM decode 48（real / circular-shuffle / geometry-only各16）で、全`seconds`は正だった。
- 保守的p10-throughput fold外挿は`4.516838993 h`、p50外挿は`3.525366874 h`。runtime gate `<=8.5 h`をPASSした。
- peak GPU memoryは`1.924051762 GB`で、memory gate `<=14 GB`をPASSした。`outer_valid_truth_access_count=0`、`trained_stage_a_model_count=0`も確認した。
- selection manifest SHAは`71358ae2725586e78ceb18cd4d19bd0ffb4c7df1fac32a9c86d73cae24d6b937`、measurement SHAは`c49a001dd2a5316db31df2022a1d0207610866a14d47e062516f80c8444b5fab`で、report内記録と実ファイルSHAが一致した。
- report SHAは`401d98f2cdc9ced437d66fc02bbe49b9287d4772e4d9036719c573a90b785c59`。package Notebook SHAは`71e37ae8b3defd8d0732913f59237b5d214b1afbccb513471d8c32bdcd667dd9`、metadata SHAは`ae10005dd26d23e9e3225425338690385c02efd06a6efe05cc25316d09dba4af`、Kaggle log SHAは`d0fba4651e41145c89ac8e28ff2c6345815b91cb8794a2880afbaf042bb1f549`。
- report SHAの固定と80行の実測検証が必要だったため、outputを`kaggle/output/stage0_v1/`へ取得した。実行済みStage 0 approvalは消費済みとして`selected_stage=implementation_only`、`kaggle_push_approved=false`へ戻し、再pushをfail-closedにした。
- Gate decisionは`request_separate_stage_a_approval`。Stage A予定規模は`1 architecture × fold 0 × seed 42 = 1 neural model`、LightGBM config 0、booster 0、PF/Beam 0、parent/control再学習0のままであり、別承認なしには開始しない。

## 2026-07-21 Stage A実行承認

- ユーザー依頼「Stage Aを開始してください」を、fold 0のpackage、Kaggle T4 push、開始確認の承認として扱った。Stage B、inference、submissionは承認範囲外である。
- 実行規模はactive variant 1、architecture 1、fold 1（fold 0のみ）、seed 42、neural model 1。LightGBM config 0、booster 0、PF/Beam run 0、parent/control再学習0。
- 比較baselineは保存済みexp209 prediction/cacheを参照し、exp209、exp221、parent exp295、circular-shuffle、geometry-only用の別modelは再学習しない。controlsはStage Aで学習する同一modelのfreeze後decodeだけである。
- Stage 0 report SHA`401d98f2cdc9ced437d66fc02bbe49b9287d4772e4d9036719c573a90b785c59`と実ファイルSHAを再照合した。保守的fold外挿`4.516839 h`、peak`1.924052 GB`のPASSを前提にする。
- source preflightでexp209 baseline cache`exp209_exp072_exp205_joint_exact_parity_fast_cache_generation_amerhu_exact_hmm_smoother_default_train_features.csv.gz`とexp115 assignment`exp115_hidden_like_spatial_holdout_from_ppt_fold_assignments.csv`のKaggle output存在を確認した。
- Stage A専用kernel id/titleは`kentookumura/exp331-prefix-gr-unary-local-ce-exact-ssm-stage-a` / `exp331 prefix gr unary local ce exact ssm stage a`、acceleratorは`NvidiaTeslaT4`に固定する。
- package bootstrapを復号し、`stage_a_fold0`、Stage 0 gate/SHA、1 model/1 fold/0 booster、Stage B/inference/submission未承認、T4を確認した。package Notebook SHAは`57079fc8891d2a74d947263175a5b038e968a158790697a25b6298adbf10b0bb`、metadata SHAは`f8cfa6ec2da0e17fb711cb6ad5a345bd45161a435dcc69972354cac7eb5924dc`。
- Stage A kernel version 1をpushした。pull-back metadataはid/title一致、`id_no=128114450`、`enable_gpu=true`、`machine_shape=NvidiaTeslaT4`、competition/exp209/exp115 source一致。`2026-07-21T11:39:54Z`時点のstatusは`RUNNING`。

## 2026-07-22 Stage A実行結果

- Kaggle kernel `kentookumura/exp331-prefix-gr-unary-local-ce-exact-ssm-stage-a` version 1は`COMPLETE`。summary生成時刻は`2026-07-21T15:47:59.216758+00:00`。
- 1 architecture × fold 0 × seed 42の1 neural modelを8 epochs学習し、early-stop local CEが最良のepoch 7を採用した。fit / early-stop / validは`556 / 62 / 155 wells`、学習時間`1,220.349 sec`、Stage A全体`14,815.790 sec = 4.115497 h`、peak GPU memory`1.889884 GB`でcompute条件をPASSした。
- real GR RMSEは`24.760360`、保存済みexp209は`12.671087`で`+12.089273 ft`悪化した。geometry-only `32.465002`には`7.704642 ft`、circular shuffle `57.878820`には`33.118460 ft`勝ち、real NLLはshuffleより`4.317422`、within10 massは`+0.428913`改善した。
- well RMSE p95はreal `44.560719`対exp209 `26.301518`で`+18.259200 ft`、worst-well regressionは`+63.109520 ft`。exp209より改善17 wells、悪化138 wells、well別delta中央値`+9.831584 ft`だった。
- distance 1000+は`26.016535`対exp209 `13.878414`、hidden-like spatialは`25.234291`対`12.761284`、hidden-like typewell-purgedは`24.169723`対`12.046808`で、全stress scopeが約`+12.1--12.5 ft`悪化した。
- finite prediction、runtime、memory、prefix clamp、real-vs-shuffle NLL、real-vs-geometry RMSE、real-vs-shuffle within10、target-in-gridはPASS。`real_rmse_vs_exp209`、`well_p95_non_regression`、`worst_well_regression`はFAILした。
- truth-freezeは`outer_valid_truth_access_count_before_freeze=0`、hidden-like assignmentはprediction freeze後読込、forbidden neighbor sources 0。780,457 rows / 155 wellsのfrozen predictionとvalidation readoutを確認した。
- summary SHAは`273e51100babeab3554a56d8853e345b1d993ced3c0d23d9568cf60e07dd3356`、model SHAは`e9cd7404eabf9192a3026184bdffb2de3f585861aa8b0dee2edd777d542fc61b`、frozen predictionのgzip / decompressed SHAは`120ae26...a82af` / `4fdee845...b844`。summary内の全12 file SHA、model、emission manifest、frozen prediction展開後SHAが実ファイルと一致した。
- 科学gate総合は`false`、decisionは`close_stage_b_without_exp331_rescue_grid`。承認済みStage A scopeを消費済みとして`selected_stage=implementation_only`、push/GPU approval flagをfalseへ戻した。Stage B、推論、提出は未実行のまま閉鎖する。
- local CE unaryはshuffle/geometryより良いGR signalを持つが、exp209より138/155 wellsで悪化しtailも壊したため、global alignment品質の代替にはならない。exp331内のarchitecture/loss/band/temperature/view/epoch救済は行わない。

## コマンドログ

```text
make new-steering EXP=exp331_prefix_gr_unary_local_ce_exact_ssm
make new-exp EXP=exp331_prefix_gr_unary_local_ce_exact_ssm
.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp331_prefix_gr_unary_local_ce_exact_ssm --notebook train --kernel-id kentookumura/exp331-prefix-gr-unary-local-ce-exact-ssm-stage0 --title "exp331 prefix gr unary local ce exact ssm stage0" --run-on-push --strict
kaggle kernels push -p experiments/exp331_prefix_gr_unary_local_ce_exact_ssm/kaggle/train --accelerator NvidiaTeslaT4
kaggle kernels logs kentookumura/exp331-prefix-gr-unary-local-ce-exact-ssm-stage0
kaggle kernels output kentookumura/exp331-prefix-gr-unary-local-ce-exact-ssm-stage0 -p experiments/exp331_prefix_gr_unary_local_ce_exact_ssm/kaggle/output/stage0_v1
.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp331_prefix_gr_unary_local_ce_exact_ssm --notebook train --kernel-id kentookumura/exp331-prefix-gr-unary-local-ce-exact-ssm-stage-a --title "exp331 prefix gr unary local ce exact ssm stage a" --run-on-push --strict
kaggle kernels push -p experiments/exp331_prefix_gr_unary_local_ce_exact_ssm/kaggle/train --accelerator NvidiaTeslaT4
kaggle kernels pull kentookumura/exp331-prefix-gr-unary-local-ce-exact-ssm-stage-a -p /tmp/exp331-stage-a-identity-v1 -m
kaggle kernels status kentookumura/exp331-prefix-gr-unary-local-ce-exact-ssm-stage-a
kaggle kernels logs kentookumura/exp331-prefix-gr-unary-local-ce-exact-ssm-stage-a
kaggle kernels output kentookumura/exp331-prefix-gr-unary-local-ce-exact-ssm-stage-a -p experiments/exp331_prefix_gr_unary_local_ce_exact_ssm/kaggle/output/stage_a_v1
```

## 再現性メモ

- seed policy: seed 42 + stable SHA256 per well/fold/view/control
- stochastic components: CUDA convolution、AdamW、dropout、dataloader order（実装時）
- deterministic anchor: false
- SHA: input/fold/view/preprocessing/model/unary/posterior/prediction/package/kernelを段階ごとに記録する設計
- gzip: decompressed content SHAを主証拠にする

## 次のアクション

1. exp331はStage A科学FAILとして閉鎖済み。Stage B、推論、提出を実行しない。
2. 代替設計exp332はexp331 closeという先行条件だけ成立した。実装・実行は別のユーザー判断がある場合だけ行う。
