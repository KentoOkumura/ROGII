# exp191_typewell_late_range_continuity_selector_on_exp176 セッションノート

## 2026-07-04 実装

- ユーザー依頼により `typewell_late_range_continuity_selector_on_exp176` backlog の実装を開始。
- `docs/legacy/steering/20260704-exp191-typewell-late-range-continuity-selector-on-exp176/` を作成。
- `experiments/exp191_typewell_late_range_continuity_selector_on_exp176/` を exp158 から作成。
- 既存 `exp190_denoised_calibrated_matching_features_on_exp148` との番号衝突を確認したため、typewell continuity 実験は exp191 とした。
- helper を `typewell_late_range_continuity_selector_on_exp176.py` にリネーム。
- train / inference notebook 起点として `exp191_typewell_late_range_continuity_selector_on_exp176_{train,inference}.py` を追加。

## 実装内容

- exp158 の Viterbi continuity selector を維持。
- parent score surface を exp157 から exp176 v3 saved boosters に差し替え。
- exp176 と同じ raw train typewell context feature を復元。
  - row-level `tlp_`
  - candidate-long `candidate_tlp_`
- exp176 v3 と同じ memory fix / feature contract に合わせ、long model 用 row feature から `tlp_` を除外。
- `likpf_mean_single`、`exp176_error_ranker_rowwise`、Viterbi variants、oracle を比較する。
- direct replacement、blend、postprocess、hard invalid、clip、submission は作らない。

## Kaggle train push 前コスト確認

- Runtime: CPU (`enable_gpu=false`)
- active posthoc audit 数: 1
- 新規 LightGBM model family 数: 0
- 新規 fold 数: 0
- 新規 booster 数: 0
- exp176 saved booster inference: 15 boosters
- Viterbi variants: 180
- exp176 control / parent 再学習: なし
- PF/Beam 再生成: なし
- submission / inference port: なし

## 検証

```bash
python3 -m py_compile \
  experiments/exp191_typewell_late_range_continuity_selector_on_exp176/typewell_late_range_continuity_selector_on_exp176.py \
  experiments/exp191_typewell_late_range_continuity_selector_on_exp176/settings.py \
  experiments/exp191_typewell_late_range_continuity_selector_on_exp176/exp191_typewell_late_range_continuity_selector_on_exp176_train.py \
  experiments/exp191_typewell_late_range_continuity_selector_on_exp176/exp191_typewell_late_range_continuity_selector_on_exp176_inference.py
.venv/bin/ruff check \
  experiments/exp191_typewell_late_range_continuity_selector_on_exp176/typewell_late_range_continuity_selector_on_exp176.py \
  experiments/exp191_typewell_late_range_continuity_selector_on_exp176/settings.py \
  experiments/exp191_typewell_late_range_continuity_selector_on_exp176/exp191_typewell_late_range_continuity_selector_on_exp176_train.py \
  experiments/exp191_typewell_late_range_continuity_selector_on_exp176/exp191_typewell_late_range_continuity_selector_on_exp176_inference.py --select F821,F401
.venv/bin/ruff format --check \
  experiments/exp191_typewell_late_range_continuity_selector_on_exp176/typewell_late_range_continuity_selector_on_exp176.py \
  experiments/exp191_typewell_late_range_continuity_selector_on_exp176/settings.py \
  experiments/exp191_typewell_late_range_continuity_selector_on_exp176/exp191_typewell_late_range_continuity_selector_on_exp176_train.py \
  experiments/exp191_typewell_late_range_continuity_selector_on_exp176/exp191_typewell_late_range_continuity_selector_on_exp176_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp191_typewell_late_range_continuity_selector_on_exp176/exp191_typewell_late_range_continuity_selector_on_exp176_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp191_typewell_late_range_continuity_selector_on_exp176/exp191_typewell_late_range_continuity_selector_on_exp176_inference.py
python3 -m json.tool \
  experiments/exp191_typewell_late_range_continuity_selector_on_exp176/exp191_typewell_late_range_continuity_selector_on_exp176_train.ipynb
python3 -m json.tool \
  experiments/exp191_typewell_late_range_continuity_selector_on_exp176/exp191_typewell_late_range_continuity_selector_on_exp176_inference.ipynb
uv run python scripts/validate_experiment.py --experiment exp191_typewell_late_range_continuity_selector_on_exp176
```

- result: PASS

## Kaggle package prepare

```bash
make prepare-kaggle-notebooks EXP=exp191_typewell_late_range_continuity_selector_on_exp176 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp191-typewell-late-continuity-train --title 'exp191 typewell late continuity train' --run-on-push --strict"
make prepare-kaggle-notebooks EXP=exp191_typewell_late_range_continuity_selector_on_exp176 EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp191-typewell-late-continuity-inference --title 'exp191 typewell late continuity inference' --strict"
python3 -m py_compile \
  experiments/exp191_typewell_late_range_continuity_selector_on_exp176/kaggle/train/typewell_late_range_continuity_selector_on_exp176.py \
  experiments/exp191_typewell_late_range_continuity_selector_on_exp176/kaggle/train/exp191_typewell_late_range_continuity_selector_on_exp176_train.py \
  experiments/exp191_typewell_late_range_continuity_selector_on_exp176/kaggle/train/settings.py \
  experiments/exp191_typewell_late_range_continuity_selector_on_exp176/kaggle/inference/exp191_typewell_late_range_continuity_selector_on_exp176_inference.py \
  experiments/exp191_typewell_late_range_continuity_selector_on_exp176/kaggle/inference/settings.py
```

- result: PASS
- train package: `experiments/exp191_typewell_late_range_continuity_selector_on_exp176/kaggle/train`
- inference package: `experiments/exp191_typewell_late_range_continuity_selector_on_exp176/kaggle/inference`
- train kernel id: `kentookumura/exp191-typewell-late-continuity-train`
- inference kernel id: `kentookumura/exp191-typewell-late-continuity-inference`
- metadata `enable_gpu`: false
- metadata `enable_internet`: false
- train `run_on_push`: true
- kernel sources:
  - `kentookumura/exp099-pf-multiobs-likelihood-train`
  - `kentookumura/exp072-exp063-full-replay-feature-cache-train`
  - `kentookumura/exp176-typewell-late-candidate-prior-train`
- generated package config confirms `planned_boosters=0`, `saved_parent_boosters_used=15`, and output prefix `exp191_typewell_late_range_continuity_selector_on_exp176`.

## 2026-07-04 Kaggle train v1 push

```bash
make push-kaggle-train EXP=exp191_typewell_late_range_continuity_selector_on_exp176
kaggle kernels pull kentookumura/exp191-typewell-late-continuity-train -p /tmp/kaggle-pull/exp191-typewell-late-continuity-train -m
kaggle kernels status kentookumura/exp191-typewell-late-continuity-train
kaggle kernels logs kentookumura/exp191-typewell-late-continuity-train
```

- push: PASS
- Kaggle kernel version: v1
- URL: https://www.kaggle.com/code/kentookumura/exp191-typewell-late-continuity-train
- metadata pull: PASS
- metadata `id_no`: 125909716
- metadata `enable_gpu`: false
- metadata `enable_internet`: false
- metadata kernel sources:
  - `kentookumura/exp072-exp063-full-replay-feature-cache-train`
  - `kentookumura/exp099-pf-multiobs-likelihood-train`
  - `kentookumura/exp176-typewell-late-candidate-prior-train`
- status after push: `KernelWorkerStatus.RUNNING`
- initial CLI logs: empty. This is expected for a running Kaggle notebook in this environment and is not treated as failure.

## 2026-07-04 Kaggle train v1 completion

```bash
kaggle kernels status kentookumura/exp191-typewell-late-continuity-train
kaggle kernels logs kentookumura/exp191-typewell-late-continuity-train
kaggle kernels output kentookumura/exp191-typewell-late-continuity-train \
  -p experiments/exp191_typewell_late_range_continuity_selector_on_exp176/kaggle/output/train_v1
```

- status: `KernelWorkerStatus.COMPLETE`
- output download: PASS
- output dir: `experiments/exp191_typewell_late_range_continuity_selector_on_exp176/kaggle/output/train_v1`
- summary status: `completed_train_side_audit`
- runtime seconds: 19,747.140143
- rows / wells: 3,783,989 / 773
- Viterbi variants: 180
- exp176 resolved saved models: 15
- typewell late-range feature count: 77

### Best result

- best variant: `viterbi_sw400_bias000_jw050_jf025_d075_std999999_md0000_seg012`
- RMSE: 10.598006879875323
- MAE: 6.402336928494969
- within10: 0.7931106565056082
- oracle label accuracy: 0.2650507176421496
- params: `switch_penalty=40.0`, `jump_penalty_weight=0.5`, `jump_free_ft=25.0`, `max_abs_delta_vs_default=75.0`, `min_segment_len=12`
- selection distribution:
  - `pf_ancc`: 1,551,994 rows / 0.410148
  - `beam_mean`: 186,204 rows / 0.049208
  - `likpf_mean`: 1,364,479 rows / 0.360593
  - `sc_ens`: 321 rows / 0.000085
  - `hyb`: 219 rows / 0.000058
  - `tvt_dense`: 83,414 rows / 0.022044
  - `tvt_densew`: 277,571 rows / 0.073354
  - `tvt_dense50`: 319,787 rows / 0.084511

### Comparison

- `likpf_mean_single`: RMSE 11.594897672217703 / MAE 7.067632584311985 / within10 0.772807479091509
- exp176 row-wise `lgb_candidate_error_ranker`: RMSE 10.641296370660122 / MAE 6.43455877101519 / within10 0.7918154624656678
- exp158 best Viterbi: RMSE 10.789163253079206
- delta vs `likpf_mean_single`: -0.9968907923423806
- delta vs exp176 row-wise: -0.04328949078479738
- delta vs exp158 best Viterbi: -0.19115637320388146

### Continuity / bucket readout

- best Viterbi path switches: 3,620 / 0.956662 per 1000 rows
- exp176 row-wise path switches: 261,391 / 69.078161 per 1000 rows
- best max path switch per 1000 rows by well: 4.045605
- exp176 row-wise max path switch per 1000 rows by well: 330.841857
- by-well vs exp176 row-wise: 417 improved / 356 worse / 0 same
- mean / median by-well delta RMSE vs exp176 row-wise: -0.075572 / -0.019083
- max regression: +1.450447 RMSE on well `6d1d74e1`
- max improvement: -10.746658 RMSE on well `57f05c51`
- distance bucket delta vs exp176 row-wise:
  - `000_050`: +0.027755 RMSE
  - `050_100`: +0.014363 RMSE
  - `100_250`: +0.017168 RMSE
  - `250_500`: +0.064221 RMSE
  - `500_1000`: +0.001444 RMSE
  - `1000_plus`: -0.051216 RMSE

### SHA / reproducibility

- exp099 source decompressed SHA: `1939d536b1e56f7c0ea3847cc386ef769b0d33759d16e816c9ce180f0532df9a`
- exp072 source decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- exp176 feature schema SHA: `f4643a36903f0a84b4f2e1f205581fcd4d6dfe18474372848a188c921126b612`
- exp176 model manifest SHA: `7f49128b9404898038765da8b48e7f687e60f40e4c69fff34e63eede17054845`
- metrics SHA: `62b22ae67db50e13ede3e040f37faf7bbf969088a8fb613e09e5d06c6103fe9e`
- bucket metrics SHA: `731169a6ba9d8b08fd00aed22b4f2dbbdff4c0391fcbeb0e32a5028902bcbb12`
- by-well SHA: `0d7f964806a5e101bf881c388b4097039c3d2af6c944e8e20926212860aff022`
- selection distribution SHA: `eb5505839bbccfe4a09508b0190ef90c2315813bfc578764f9ecef9ab84d313c`
- score summary SHA: `f350fb2f4a115450146ef93f943d73f93e23632effe26eaaaae4679003eaffc2`
- Viterbi params SHA: `aad315a1f3cd1130203e570f00d1a2fe60e03beb995250cc9653638508400edc`
- OOF prediction decompressed SHA: `c940900bbd72ce5b3410a19dc47096780b1008e6df9d99a32c360657e0a7317c`
- best Viterbi prediction SHA: `1a6955b5cae23e283c8ede4328427adbd00032273936aae6b6ecbb7b13893dfa`

### Judgment

exp176 row-wise より global RMSE と path continuity は改善した。特に path switch は 261,391 から 3,620 まで減ったため、continuity audit としては supported。ただし near / mid distance buckets は小幅悪化し、356 wells は exp176 row-wise から悪化したため、direct selected TVT inference / submission には進めない。後続で使う場合は exp148 系への confidence / segment-stability feature surface に限定する。
