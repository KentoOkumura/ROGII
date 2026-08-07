# exp177_beam_topk_bimodal_gate_posthoc_audit セッションノート

## 目的

`beam_topk_bimodal_gate_posthoc_audit` バックログを実装し、exp173 の保存済み Beam top-K diagnostics を使って、二峰性条件がある row だけ posterior / top2 / weighted mean へ置換する no-training posthoc gate を train-side で監査できる状態にする。

## 現在の状態

- Route: `pf_beam`
- 状態: completed_train_side_rejected_no_submit
- CV: best gated policy RMSE 11.837783911
- LB: なし
- inference / submit: 対象外

## Push 前の計算規模

- active Beam regeneration variants: 0
- active posthoc gate audit grid: 1
- LightGBM config 数: 0
- fold 数: 0
- 合計 booster 数: 0
- control / 親実験再学習: なし
- GPU: 不使用

## コマンドログ

### 2026-07-03 実装

```bash
make new-steering EXP=exp177_beam_topk_bimodal_gate_posthoc_audit
make new-exp EXP=exp177_beam_topk_bimodal_gate_posthoc_audit
.venv/bin/python -m py_compile experiments/exp177_beam_topk_bimodal_gate_posthoc_audit/beam_topk_bimodal_gate_posthoc_audit.py experiments/exp177_beam_topk_bimodal_gate_posthoc_audit/exp177_beam_topk_bimodal_gate_posthoc_audit_train.py experiments/exp177_beam_topk_bimodal_gate_posthoc_audit/exp177_beam_topk_bimodal_gate_posthoc_audit_inference.py
.venv/bin/ruff check experiments/exp177_beam_topk_bimodal_gate_posthoc_audit/beam_topk_bimodal_gate_posthoc_audit.py experiments/exp177_beam_topk_bimodal_gate_posthoc_audit/exp177_beam_topk_bimodal_gate_posthoc_audit_train.py experiments/exp177_beam_topk_bimodal_gate_posthoc_audit/exp177_beam_topk_bimodal_gate_posthoc_audit_inference.py --select F821,F401,E501
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp177_beam_topk_bimodal_gate_posthoc_audit/exp177_beam_topk_bimodal_gate_posthoc_audit_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp177_beam_topk_bimodal_gate_posthoc_audit/exp177_beam_topk_bimodal_gate_posthoc_audit_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp177_beam_topk_bimodal_gate_posthoc_audit/exp177_beam_topk_bimodal_gate_posthoc_audit_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp177_beam_topk_bimodal_gate_posthoc_audit/exp177_beam_topk_bimodal_gate_posthoc_audit_inference.py
make validate-exp EXP=exp177_beam_topk_bimodal_gate_posthoc_audit
make prepare-kaggle-notebooks EXP=exp177_beam_topk_bimodal_gate_posthoc_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp177-beam-topk-bimodal-gate-posthoc-audit-train --title 'exp177 beam topk bimodal gate posthoc audit train' --run-on-push --strict"
```

- `.steering/20260703-exp177-beam-topk-bimodal-gate-posthoc-audit/` を作成。
- `experiments/exp177_beam_topk_bimodal_gate_posthoc_audit/` を作成。
- `beam_topk_bimodal_gate_posthoc_audit.py` を追加。
- `config.yaml` を PF/Beam route / exp173 output posthoc audit に更新。
- Jupytext 起点の train / inference script を追加。inference は no-submission guard。
- `py_compile`: PASS。
- ruff `F821,F401,E501`: PASS。
- Jupytext train / inference 変換と `--test`: PASS。
- `make validate-exp`: strict PASS。
- Kaggle train package 生成: PASS。
- train kernel metadata:
  - id: `kentookumura/exp177-beam-topk-bimodal-gate-posthoc-audit-train`
  - title: `exp177 beam topk bimodal gate posthoc audit train`
  - GPU: false
  - internet: false
  - run_on_push: true
  - kernel source: `kentookumura/exp173-beam-topk-path-posterior-audit-train`

### 2026-07-03 Kaggle train v1 実行

push 前確認:

- 実行対象: train-side deterministic posthoc gate audit
- active Beam regeneration variants: 0
- active posthoc gate audit grid: 1
- LightGBM config 数: 0
- fold 数: 0
- booster 数: 0
- GPU: false
- internet: false
- control / parent 再学習: なし

```bash
make push-kaggle-train EXP=exp177_beam_topk_bimodal_gate_posthoc_audit
kaggle kernels pull kentookumura/exp177-beam-topk-bimodal-gate-posthoc-audit-train -p /tmp/kaggle-pull/exp177-beam-topk-bimodal-gate-posthoc-audit-train -m
kaggle kernels logs kentookumura/exp177-beam-topk-bimodal-gate-posthoc-audit-train
kaggle kernels status kentookumura/exp177-beam-topk-bimodal-gate-posthoc-audit-train
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp177-beam-topk-bimodal-gate-posthoc-audit-train
```

- Kaggle train kernel version 1 を push 成功。
- URL: `https://www.kaggle.com/code/kentookumura/exp177-beam-topk-bimodal-gate-posthoc-audit-train`
- `kaggle kernels pull -m` で同 kernel id の存在確認は成功。
- `kaggle kernels status`: `KernelWorkerStatus.RUNNING`
- 通常 `kaggle kernels logs` は空出力。実行中 logs が空になる既知挙動として扱い、失敗判定や再 push はしない。
- `logs -f` は開始後にユーザー指示で停止。監視を止め、Kaggle notebook 実行自体は継続中。

### 2026-07-03 Kaggle train v1 完了確認

ユーザーから完了連絡後、同じ kernel id で確認した。

```bash
kaggle kernels status kentookumura/exp177-beam-topk-bimodal-gate-posthoc-audit-train
kaggle kernels logs kentookumura/exp177-beam-topk-bimodal-gate-posthoc-audit-train
kaggle kernels output kentookumura/exp177-beam-topk-bimodal-gate-posthoc-audit-train -p experiments/exp177_beam_topk_bimodal_gate_posthoc_audit/kaggle/output/train_v1
```

- status: `KernelWorkerStatus.COMPLETE`
- output: `experiments/exp177_beam_topk_bimodal_gate_posthoc_audit/kaggle/output/train_v1`
- runtime: 409.031 sec
- rows / wells: 3,783,989 / 773
- baseline `likpf_mean`: RMSE 11.594897884 / MAE 7.067632615 / within10 0.772807479
- best policy: `beam_topk_sm11_bw64__and_sep_ge_q90__cost_le_q10__replace_posterior_mean_t1`
  - RMSE 11.837783911
  - MAE 7.299786632
  - within10 0.760339948
  - delta RMSE vs baseline +0.242886027
  - changed rows 384,695 / 71 wells
  - changed subset baseline RMSE 10.269740849
  - changed subset candidate RMSE 12.706185740
  - changed subset delta +2.436444890
  - near `000_050` delta +0.027170813
  - longtail `1000_plus` delta +0.241960608
  - Beam-likPF gap top quartile delta +0.501165946
  - max well regression +22.519192863
- decision: `diagnostic_only_not_submit_candidate`
- artifact SHA:
  - policy metrics `8454de5c20e81fbb2946b4110213d3569e47e754f16d78f2a9b2611f9a03b177`
  - gate thresholds `ea547bd0386b59ef673b801bfae48819911276f9fffbd164cad3779ebdad28a7`
  - group metrics `34c7a540a1d62a0514b0e3bd0c2cc6540cb25deff5ec63bd2dafd78ddbe051b6`
  - by-well `4325525c2405ef9ae4d5347e1df87819c2909b0cc93807c407cc0a460b64e9e2`
  - summary `f640e36fcaeb6757b493c876c61458bc70e9bd4c9cdd2d834586b76577d9d8ac`
- input decompressed SHA:
  - exp173 `candidate_wide` `f993aaed3f59a39f3e367e1c18b3a7a394a254db09c1a5277d90d605621613bd`
  - exp173 `topk_diagnostics` `08b1ed91742e4352b732fb739fdd59a8b4c53f53582f8a1c295a7b123e070301`
  - exp173 `topk_paths` `cf23b20a5b2ee9c8266f6272374463ec49cf8229c78570b73908cd346f4c73cc`

結論: negative。二峰性 / low-cost-gap で絞っても posterior / top2 / weighted mean replacement は global、changed subset、near、longtail、Beam-likPF gap、worst-well をすべて壊した。inference port / submit はしない。

## 実装内容

- 入力:
  - `exp173_beam_topk_path_posterior_audit_topk_diagnostics.csv.gz`
  - `exp173_beam_topk_path_posterior_audit_topk_paths.csv.gz`
  - `exp173_beam_topk_path_posterior_audit_candidate_wide.csv.gz`
  - `exp173_beam_topk_path_posterior_audit_candidate_metrics.csv`
- Baseline: `likpf_mean`
- Replacement:
  - `top2_commit`
  - `topk_weighted_mean`
  - `posterior_mean_t1/t2/t4/t8/t16`
- Gate:
  - `top1_top2_sep >= q75/q90`
  - `top2_cost_gap_per_row <= q10/q25`
  - `topk_entropy >= q75/q90`
  - `topk_spread >= q75/q90`
  - 上記の AND 条件

## 再現性メモ

- seed policy: `deterministic_posthoc_grid_no_rng`
- stochastic components: 上流 exp072 PF/Beam/likPF cache と exp173 Beam top-K generation の固定生成物のみ
- exp177 内の model training / Beam regeneration / PF regeneration: なし
- CPU/GPU runtime: CPU、GPU 不使用
- gzip input/output は raw SHA と decompressed SHA を summary に記録する。
- model manifest / model SHA: model なし
- prediction SHA: submission candidate なし
- submission SHA: submission なし
- rerun check: train-side audit のため、positive で downstream 候補になった場合だけ追加実施する。

## 次のコマンド

なし。

## 次のアクション

1. exp177 は完了/不採用として閉じる。
2. Beam top-K posterior 系の direct replacement / confidence feature follow-up は進めない。
