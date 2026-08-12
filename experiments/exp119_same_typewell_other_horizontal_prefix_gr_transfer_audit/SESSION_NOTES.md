# exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit セッションノート

## 目的

同じ native typewell group の他 horizontal well について、source 側を疑似 predict start 前の raw GR と `TVT_input` だけに制限して、query well の evaluation-zone raw GR へ転用できる信号があるかを train pseudo-tail OOF で診断する。

## 現在の状態

- Route: ensemble
- 状態: Kaggle train v2 完了 / rejected
- CV: best `likpf_mean` RMSE 11.594897672
- LB: まだなし

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

```bash
uv run python scripts/new_steering.py --experiment exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit
uv run python scripts/new_experiment.py --name exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit --source experiments/exp109_typewell_neighbor_prior_features
```

### 実行済み

```bash
uv run python -m py_compile experiments/exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit/same_typewell_other_horizontal_prefix_gr_transfer_audit.py experiments/exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit/settings.py
uv run ruff check experiments/exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit/same_typewell_other_horizontal_prefix_gr_transfer_audit.py experiments/exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit/settings.py
uv run python scripts/validate_experiment.py --experiment exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit
EXPERIMENT_ALLOW_LOCAL=1 .venv/bin/python -c "... run_audit smoke with max_rows=20000, max_wells=12, max_source_wells=2, stride=64 ..."
make prepare-kaggle-notebooks EXP=exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp118-same-typewell-prefix-gr-transfer-train --title 'exp118 same typewell prefix gr transfer train' --run-on-push --strict"
uv run python -m py_compile experiments/exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit/kaggle/train/same_typewell_other_horizontal_prefix_gr_transfer_audit.py experiments/exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit/kaggle/train/settings.py
uv run ruff check experiments/exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit/kaggle/train/same_typewell_other_horizontal_prefix_gr_transfer_audit.py experiments/exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit/kaggle/train/settings.py
kaggle kernels push -p experiments/exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit/kaggle/train
kaggle kernels output kentookumura/exp118-same-typewell-prefix-gr-transfer-train -p experiments/exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit/kaggle/output/train_v1
make prepare-kaggle-notebooks EXP=exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp118-same-typewell-prefix-gr-transfer-train --title 'exp118 same typewell prefix gr transfer train' --run-on-push --strict"
kaggle kernels push -p experiments/exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit/kaggle/train
kaggle kernels logs kentookumura/exp118-same-typewell-prefix-gr-transfer-train
kaggle kernels output kentookumura/exp118-same-typewell-prefix-gr-transfer-train -p experiments/exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit/kaggle/output/train_v2
```

## 変更点

- `docs/legacy/steering/20260624-exp118-same-typewell-other-horizontal-prefix-gr-transfer-audit/` を作成。その後、既存 `exp118_spatial_neighbor_prior_confidence_gate_on_exp092` との番号衝突を確認し、`docs/legacy/steering/20260624-exp119-same-typewell-other-horizontal-prefix-gr-transfer-audit/` に改番した。
- `experiments/exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit/` を exp109 から作成。
- `config.yaml` を same-typewell prefix GR transfer audit に更新。
- `same_typewell_other_horizontal_prefix_gr_transfer_audit.py` を追加。
  - exp099 v2 feature cache から `well`、row id、`md_since`、base candidates、`true_tvt` を復元する。
  - raw horizontal CSV から各 well の pseudo-prefix anchor、raw GR、`TVT_input` を読む。
  - GroupKFold で valid well の source pool を train-fold wells に制限する。
  - `same_typewell_gr_match`、`same_typewell_random_control`、`different_typewell_gr_match` を同じ window / metric で比較する。
  - source prefix の offset / local slope / local path delta を `last_known_tvt` に足して prior にし、base candidates への clipped correction も保存する。
  - candidate metrics、bucket metrics、by-well metrics、signal metrics、OOF predictions、feature schema、summary JSON を保存する。
- train notebook を設定確認、入力確認、audit 実行、metrics 表示の構成に更新。
- inference notebook は train-side audit only guard に更新。
- 小さいローカル smoke は rows 20,000 / wells 5 で完走し、best candidate 名まで出力できることを確認した。これは正式 CV ではない。smoke 生成物は `artifacts/` に残っているが、`metrics.json` は pending 状態へ戻した。
- Kaggle train package を `experiments/exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit/kaggle/train/` に生成した。
- Kaggle v1/v2 実行時の generated `kernel-metadata.json` は `enable_gpu=false`、`enable_internet=false`、kernel id `kentookumura/exp118-same-typewell-prefix-gr-transfer-train`、source `kentookumura/exp099-pf-multiobs-likelihood-train` / `kentookumura/exp065-typewell-supertype-cluster-cv-audit-train`。改番後の current package metadata は `kentookumura/exp119-same-typewell-prefix-gr-transfer-train` に更新済み。
- Kaggle train v1 は push 成功したが、約 2088 秒で `DeadKernelError: Kernel died`。output は support files と log のみで、`artifacts/` は取得できなかった。原因は GR matching / candidate grid の実行量またはメモリ過大の可能性が高い。
- v2 では `group_methods` を `native_overlap_0p999` のみに絞り、`max_source_wells=6`、correction は `likpf_mean` の `alpha=0.10` / `clip=20` のみに縮小した。same-typewell GR match / same-typewell random / different-typewell GR match の negative control 比較は維持した。
- v2 package の py_compile / ruff / generated config 確認を通した。
- Kaggle train v2 を同じ kernel id `kentookumura/exp118-same-typewell-prefix-gr-transfer-train` に push 成功。Kaggle kernel version 2。
- v2 は完了。logs では rows 3,783,989 / wells 773、runtime 1149.7 sec。best は baseline `likpf_mean` のまま RMSE 11.594897672 / MAE 7.067632584 / within10 0.772807479。
- v2 output download は大きい OOF gzip で接続が切れた。candidate / bucket / by-well / feature schema は取得済み。`summary.json`、`signal_metrics.csv`、OOF gzip は未取得。
- candidate metrics の best non-baseline は `different_typewell_gr_match_slope_likpf_mean_corr_a0p1_c20` RMSE 11.607097336。same-typewell GR match best は `same_typewell_gr_match_slope_likpf_mean_corr_a0p1_c20` RMSE 11.614959308 で、`likpf_mean` より +0.020061636 悪化した。
- by-well では same-typewell slope correction が 291 wells 改善 / 407 wells 悪化 / 75 wells 同値、最大悪化 +1.907160 RMSE、最大改善 -1.903311 RMSE、平均 delta +0.120090 RMSE。

## 再現性メモ

- seed policy: deterministic well fold assignment with fixed seed 42, same-typewell random control は SHA256 由来の source center index。
- stochastic components: 新規なし。上流 exp099 / exp065 artifacts は固定入力。
- CPU/GPU runtime: CPU, GPU disabled。
- Kaggle kernel id / version: `kentookumura/exp118-same-typewell-prefix-gr-transfer-train` v1 failed / v2 completed。これは改番前の旧 kernel id。
- input / feature schema SHA: summary JSON 未取得のため input SHA は未記録。取得済み feature schema CSV SHA は `148a1df790b142e2e9731d000501349a29887e35453936d7b1cf721263ea67f3`。
- cluster assignment SHA: summary JSON 未取得のため未記録。
- feature content SHA: gzip は decompressed content SHA を主証拠にする。
- model manifest / model SHA: model なし。
- prediction SHA: OOF gzip 未取得のため未記録。candidate metrics SHA は `7ddc6a69369f122440a7a01c386576f002abcd077e2a4afd316c504b57515914`、bucket metrics SHA は `6acaa0ea97ed763fb408a74f645b1ab1a142543be41ffda6a0a882a8b7bb7198`、by-well metrics SHA は `dd583e1db065dd4dce365f645e948a5b8b478273a4010975057e1d5192341b45`。
- submission SHA: submission なし。
- rerun check: 未実行。

## 次のアクション

1. `same_typewell_other_horizontal_prefix_gr_transfer_audit` は完了として閉じる。
2. direct correction / candidate path / inference port はしない。
3. 使う場合は GR match score、coverage、source count を quality diagnostic として別の confidence / add-only feature 実験に限定する。
