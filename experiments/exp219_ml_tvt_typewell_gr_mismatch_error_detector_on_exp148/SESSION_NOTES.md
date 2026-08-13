# exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148 セッションノート

## 2026-07-07 実装

### 狙い

`backlog/KAGGLE_DIRECTION.md` の `ml_tvt_typewell_gr_mismatch_error_detector_on_exp148` backlog を実装する。exp148 の ML 予測 TVT を typewell TVT 軸上の仮位置として、horizontal GR と typewell GR の局所 window 類似度を offset 探索し、high-error row を検出できるか確認する。

### 実装方針

- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- route: `ml_model`
- 初期 variant: no-training OOF readout
- control retraining: なし
- LightGBM training: なし
- inference / submit: 初期実装では対象外

### Feature

`ml_tvt_typewell_gr_mismatch_error_detector_on_exp148.py` で以下を生成する。

- `pred_tvt + offset` の offset は `[-50, -25, -10, 0, 10, 25, 50]` ft。
- 各 offset で horizontal GR window と typewell GR window を比較し、window RMSE、NCC、derivative NCC、missing rate から score を計算する。
- summary feature は `score_at_ml`、`best_offset`、`best_score`、`score_gap`、`entropy`、`decoy_gap`、`raw_vs_denoised_score_gap`、`local_z_mse`、`abs(best_offset) x md_since`、`score_gap x candidate_disagreement`。
- optional に exp145 learned-likelihood feature cache から `candidate_tvt_range` / `candidate_tvt_std` / `learned_prob_entropy` を target-free disagreement として join する。

leakage 防止として、`target_tvt`、`abs_error`、`abs_error_gt*`、true-error rank、oracle best は feature source に使わない。これらは readout label として feature cache に role 明記して保存する。

### Kaggle train 実行前確認

- active variants: 0
- active modes: 0
- LightGBM configs: 0
- folds: 0
- planned boosters: 0
- train notebook split: なし
- control / parent retraining: なし
- GPU: disabled

### 再現性メモ

- Feature generation に乱数は使わない。
- shuffled typewell decoy は well id から SHA256 で決まる deterministic roll。
- exp148 OOF prediction と optional exp145 feature cache は固定 artifact として読み、gzip は decompressed SHA を記録する。
- deterministic submission anchor ではない。初期実装では `submission.csv` を作らない。

### 検証予定

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148_inference.py
.venv/bin/python -m py_compile experiments/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148/ml_tvt_typewell_gr_mismatch_error_detector_on_exp148.py experiments/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148_train.py experiments/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148_inference.py experiments/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148/settings.py
.venv/bin/ruff check experiments/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148/ml_tvt_typewell_gr_mismatch_error_detector_on_exp148.py experiments/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148_train.py experiments/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148_inference.py experiments/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148/settings.py --select F821,F401
make validate-exp EXP=exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148
```

### 検証結果

- `py_compile`: pass
- `ruff --select F821,F401`: pass
- Jupytext train / inference convert: pass
- Jupytext train / inference `--test`: pass
- `make validate-exp EXP=exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148`: pass
- ローカル実データ smoke: 未実行。ローカル workspace に exp148 train v1 prediction artifact と exp145 train v2 feature artifact が存在しないため、実行確認は Kaggle Notebook 上で行う。

### Kaggle package prepare

```bash
make prepare-kaggle-notebooks EXP=exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp219-ml-tvt-gr-mismatch-exp148-train --title 'exp219 ml tvt gr mismatch exp148 train' --run-on-push --strict"
make prepare-kaggle-notebooks EXP=exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148 EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp219-ml-tvt-gr-mismatch-exp148-inference --title 'exp219 ml tvt gr mismatch exp148 inference' --strict"
.venv/bin/python -m py_compile experiments/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148/kaggle/train/ml_tvt_typewell_gr_mismatch_error_detector_on_exp148.py experiments/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148/kaggle/train/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148_train.py experiments/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148/kaggle/train/settings.py experiments/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148/kaggle/inference/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148_inference.py experiments/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148/kaggle/inference/settings.py
```

- train package: `experiments/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148/kaggle/train`
- inference package: `experiments/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148/kaggle/inference`
- train kernel id: `kentookumura/exp219-ml-tvt-gr-mismatch-exp148-train`
- inference kernel id: `kentookumura/exp219-ml-tvt-gr-mismatch-exp148-inference`
- train metadata: CPU, internet off, `run_on_push=true`, kernel sources `kentookumura/exp148-train` and `kentookumura/exp145-learned-likelihood-rawtest-feature-generator-parity-train`
- inference metadata: CPU, internet off, `run_on_push=false`, no submission generation
- package py_compile: pass

### Kaggle train push v1

```bash
kaggle kernels push -p experiments/exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148/kaggle/train
kaggle kernels pull kentookumura/exp219-ml-tvt-gr-mismatch-exp148-train -p /tmp/kaggle-pull/exp219-ml-tvt-gr-mismatch-exp148-train -m
kaggle kernels status kentookumura/exp219-ml-tvt-gr-mismatch-exp148-train
kaggle kernels logs kentookumura/exp219-ml-tvt-gr-mismatch-exp148-train
```

- pushed version: 1
- Kaggle URL: https://www.kaggle.com/code/kentookumura/exp219-ml-tvt-gr-mismatch-exp148-train
- push warning: Kaggle rejected kernel source `kentookumura/exp145-learned-likelihood-rawtest-feature-generator-parity-train` as invalid, so the pulled metadata contains only `kentookumura/exp148-train` plus competition data.
- impact: exp145 candidate disagreement is optional; if the artifact is unavailable, the readout records `candidate_disagreement_available=false` and continues. exp115 hidden-like subgroup is also optional and may be unavailable unless the artifact is present in another source.
- existence check: `kaggle kernels pull ... -m` succeeded; pulled metadata id_no `126258587`.
- initial status: `KernelWorkerStatus.RUNNING`
- initial logs: empty while running; this is expected for Kaggle CLI logs in this environment and is not a failure signal.
- recheck after 2 minutes: status remains `KernelWorkerStatus.RUNNING`; logs still empty.

### Kaggle train v1 完了確認

```bash
kaggle kernels status kentookumura/exp219-ml-tvt-gr-mismatch-exp148-train
kaggle kernels logs kentookumura/exp219-ml-tvt-gr-mismatch-exp148-train
```

- status: `COMPLETE`
- runtime: 約 1004 sec
- rows / wells / feature columns: 3,783,989 / 773 / 35
- feature schema rows: 46
- exp148 base `lgb_mean`: RMSE 8.501281182、MAE 5.335654736、within10 0.856332035
- primary signal: `mlgr_mismatch_signal`
- primary signal AUC for `abs_error_gt10`: 0.573943003
- primary high signal q90: cutoff 4.651721096、rows 378,399、wells 689、abs_error_mean 7.608582497、abs_error_lift 1.425988538、error_gt_rate 0.234519647、error_gt_lift 1.632372582、RMSE 12.139380847
- diagnostic correction best: `base_exp148_lgb_mean` のまま。RMSE 8.501281182、delta 0.0、MAE 5.335654736、within10 0.856332035
- next gate: `auc_threshold=0.65`、`error_lift_threshold=1.5`、`proceed_to_lgb_addonly=false`

### 判断

high-mismatch q90 は error_gt_lift 1.63 で誤差濃縮はあるが、primary AUC は 0.574 で採用目安 0.65 に届かない。`best_offset` を使う補正診断も exp148 base を更新しなかった。

したがって exp219 は no-training readout として完了/不採用にする。exp148/exp193 add-only LightGBM、inference、submit には進めない。残す場合は weak risk flag / bucket readout として、将来の confidence ensemble 材料に限定する。
