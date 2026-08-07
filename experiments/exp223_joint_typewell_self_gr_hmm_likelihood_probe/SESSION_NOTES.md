# exp223_joint_typewell_self_gr_hmm_likelihood_probe セッションノート

## 目的

`joint_typewell_self_gr_hmm_likelihood_probe` backlog を実装する。`exp209` exact HMM の typewell GR emission を主軸にし、visible prefix 由来 self-GR motif likelihood を弱い clipped boost として同時利用する train-side diagnostic を作る。

## 現在の状態

- Route: `ensemble`
- 状態: Kaggle train v1 完了 / train-side positive vs exp072 / direct port 不採用
- CV: 11.349950650 (`hmm_selfgr_boost_only_a070_c100`)
- LB: 未提出
- GPU cost: なし。CPU-only HMM feature generation audit。
- Booster count: 0
- Parent/control retraining: なし
- Inference / submit: なし

## 実装メモ

- 親は `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`。
- exp072 full replay cache は再生成せず、保存済み exp072 cache を比較基準として読む。
- `exact_hmm_smoother.py` に self-GR descriptor matching surface を追加。
- self-GR surface は raw GR と finite `TVT_input` prefix だけから作る。unknown-suffix true TVT、OOF absolute error、oracle best、true-error rank は使わない。
- active variants は初回 runtime-limited run として `alpha=[0.07, 0.15]` x `clip=[1.0]` x `mode=[boost_only]` = 2。
- model/config/fold/booster count は 0。

## コマンドログ

```bash
python3 scripts/new_steering.py --experiment exp223_joint_typewell_self_gr_hmm_likelihood_probe
python3 scripts/new_experiment.py --name exp223_joint_typewell_self_gr_hmm_likelihood_probe --source experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation
```

- result: scaffold 作成済み。

```bash
.venv/bin/python -m py_compile \
  experiments/exp223_joint_typewell_self_gr_hmm_likelihood_probe/settings.py \
  experiments/exp223_joint_typewell_self_gr_hmm_likelihood_probe/exact_hmm_smoother.py \
  experiments/exp223_joint_typewell_self_gr_hmm_likelihood_probe/feature_cache.py \
  experiments/exp223_joint_typewell_self_gr_hmm_likelihood_probe/direct_hmm_comparison.py \
  experiments/exp223_joint_typewell_self_gr_hmm_likelihood_probe/joint_cache_generation.py \
  experiments/exp223_joint_typewell_self_gr_hmm_likelihood_probe/exp072_feature_cache.py \
  experiments/exp223_joint_typewell_self_gr_hmm_likelihood_probe/exp223_joint_typewell_self_gr_hmm_likelihood_probe_train.py \
  experiments/exp223_joint_typewell_self_gr_hmm_likelihood_probe/exp223_joint_typewell_self_gr_hmm_likelihood_probe_inference.py
.venv/bin/ruff check experiments/exp223_joint_typewell_self_gr_hmm_likelihood_probe --select F821
python3 -m json.tool experiments/exp223_joint_typewell_self_gr_hmm_likelihood_probe/metrics.json
python3 scripts/validate_experiment.py --experiment exp223_joint_typewell_self_gr_hmm_likelihood_probe
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 experiments/exp223_joint_typewell_self_gr_hmm_likelihood_probe/exp223_joint_typewell_self_gr_hmm_likelihood_probe_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 experiments/exp223_joint_typewell_self_gr_hmm_likelihood_probe/exp223_joint_typewell_self_gr_hmm_likelihood_probe_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp223_joint_typewell_self_gr_hmm_likelihood_probe/exp223_joint_typewell_self_gr_hmm_likelihood_probe_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp223_joint_typewell_self_gr_hmm_likelihood_probe/exp223_joint_typewell_self_gr_hmm_likelihood_probe_inference.py
```

- result: PASS。

## Runtime-limited 初回 run への変更

User request: 12時間以内に収めるため `alpha=[0.07, 0.15]` で進める。

- 初回 active variants: `hmm_selfgr_boost_only_a070_c100`, `hmm_selfgr_boost_only_a150_c100`
- `alpha=0.03` と `symmetric` mode は後続候補へ延期。
- feature count: 17
- model/config/fold/booster count: 0

```bash
.venv/bin/python - <<'PY'
import sys
from pathlib import Path
import yaml
exp = Path('experiments/exp223_joint_typewell_self_gr_hmm_likelihood_probe')
sys.path.insert(0, str(exp.resolve()))
from exact_hmm_smoother import prepare_self_gr_emission_variants, self_gr_variant_feature_columns
cfg = yaml.safe_load((exp / 'config.yaml').read_text())
variants, _ = prepare_self_gr_emission_variants(cfg['self_gr_emission'])
cols = self_gr_variant_feature_columns(variants)
print('variant_count', len(variants))
print('variant_names', [v['name'] for v in variants])
print('feature_count', len(cols))
print('expected_feature_count', cfg['feature_cache']['hmm']['expected_feature_count'])
PY
python3 -m json.tool experiments/exp223_joint_typewell_self_gr_hmm_likelihood_probe/metrics.json
python3 scripts/validate_experiment.py --experiment exp223_joint_typewell_self_gr_hmm_likelihood_probe
.venv/bin/python -m py_compile \
  experiments/exp223_joint_typewell_self_gr_hmm_likelihood_probe/exact_hmm_smoother.py \
  experiments/exp223_joint_typewell_self_gr_hmm_likelihood_probe/direct_hmm_comparison.py \
  experiments/exp223_joint_typewell_self_gr_hmm_likelihood_probe/joint_cache_generation.py \
  experiments/exp223_joint_typewell_self_gr_hmm_likelihood_probe/feature_cache.py
```

- result: PASS。variant_count 2、feature_count 17、expected_feature_count 17。

## 次の確認

- Kaggle train v1 を push して実行する。
- push する実行内容は CPU-only / 2 HMM variants / 0 LightGBM configs / 0 folds / 0 boosters / control retraining なし。
- Kaggle train 完了後、metrics、SHA、result、experiment_summary、KAGGLE_DIRECTION を更新する。

## Kaggle train v1 run

User request: Kaggle で実行する。

- kernel id: `kentookumura/exp223-joint-typewell-self-gr-hmm-likelihood-probe-train`
- title: `exp223 joint typewell self gr hmm likelihood probe train`
- active variants: `hmm_selfgr_boost_only_a070_c100`, `hmm_selfgr_boost_only_a150_c100`
- runtime: CPU, internet off
- model/config/fold/booster count: 0 / 0 / 0 / 0
- parent/control retraining: なし
- inference / submit: なし

```bash
python3 scripts/prepare_kaggle_notebooks.py --experiment exp223_joint_typewell_self_gr_hmm_likelihood_probe --notebook train --kernel-id kentookumura/exp223-joint-typewell-self-gr-hmm-likelihood-probe-train --title "exp223 joint typewell self gr hmm likelihood probe train" --run-on-push --strict
kaggle kernels push -p experiments/exp223_joint_typewell_self_gr_hmm_likelihood_probe/kaggle/train
kaggle kernels pull kentookumura/exp223-joint-typewell-self-gr-hmm-likelihood-probe-train -p /tmp/kaggle-pull/exp223-joint-typewell-self-gr-hmm-likelihood-probe-train -m
```

- result: FAIL
- error: `400 Client Error: Bad Request ... SaveKernel`
- long kernel pull: `403 Client Error ... GetKernel`
- interpretation: package size is small and id/title slug match, so use the same experiment folder with a shorter slug/title pair.
- recovery kernel id: `kentookumura/exp223-selfgr-hmm-train`
- recovery title: `exp223 selfgr hmm train`

```bash
python3 scripts/prepare_kaggle_notebooks.py --experiment exp223_joint_typewell_self_gr_hmm_likelihood_probe --notebook train --kernel-id kentookumura/exp223-selfgr-hmm-train --title "exp223 selfgr hmm train" --run-on-push --strict
kaggle kernels push -p experiments/exp223_joint_typewell_self_gr_hmm_likelihood_probe/kaggle/train
kaggle kernels pull kentookumura/exp223-selfgr-hmm-train -p /tmp/kaggle-pull/exp223-selfgr-hmm-train-v1 -m
kaggle kernels status kentookumura/exp223-selfgr-hmm-train
kaggle kernels logs kentookumura/exp223-selfgr-hmm-train
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp223-selfgr-hmm-train
```

- result: push PASS
- version: 1
- URL: https://www.kaggle.com/code/kentookumura/exp223-selfgr-hmm-train
- metadata pull: success, `id_no=126365791`
- status: `KernelWorkerStatus.RUNNING`
- CLI logs: empty while running
- logs follow: empty; local follow command cancelled, Kaggle notebook remains running

## Kaggle train v1 完了結果

```bash
kaggle kernels status kentookumura/exp223-selfgr-hmm-train
kaggle kernels logs kentookumura/exp223-selfgr-hmm-train > /tmp/exp223_selfgr_hmm_train_v1_logs.json
kaggle kernels output kentookumura/exp223-selfgr-hmm-train -p /tmp/kaggle-output/exp223-selfgr-hmm-train-v1
```

- status: `KernelWorkerStatus.COMPLETE`
- local output: `/tmp/kaggle-output/exp223-selfgr-hmm-train-v1`
- rows / wells: 3,783,989 / 773
- elapsed: 39,029.366 sec (約 10h50m29s)
- ok wells / skipped wells: 773 / 0
- feature content SHA (decompressed): `0eb48b5516276b0ab7b2191a52a39ebb89d9997363cc7839ede519c7863baa0c`
- joint summary SHA: `e7a3b4502a3ee4296b219be8af018568056a6c6c4319f745d848a36589ab893e`

Overall:

- best: `hmm_selfgr_boost_only_a070_c100`
- RMSE / MAE / within10: 11.349950650 / 6.471271592 / 0.794830006
- delta vs exp072 `likpf_mean`: RMSE -0.244947018、MAE -0.596360991、within10 +0.022027812
- `hmm_selfgr_boost_only_a150_c100`: RMSE 11.559594481、delta RMSE -0.035303188
- exp072 `likpf_mean`: RMSE 11.594897668、MAE 7.067632583、within10 0.772802194

Bucket / subgroup:

- Distance bucket delta RMSE vs exp072 `likpf_mean`: 000_050 -0.283719、050_100 -0.412821、100_250 -0.196443、250_500 -0.179257、500_1000 -0.396934、1000_plus -0.247524。
- Hidden-like delta RMSE vs exp072 `likpf_mean`: verification_like_spatial -1.180418、verification_like_typewell_purged -1.240497。
- By-well: 461 improved / 312 worsened。median delta -0.549381、mean delta -0.534391。
- Worst regression: `b19b0395` +46.954683 RMSE。次点 `97cd5bf9` +36.021001、`8a3da6d1` +35.692945。
- Best improvement: `86454a6f` -53.706814。
- Step delta: `hmm_selfgr_boost_only_a070_c100` mean 0.010138、p99 0.061。exp072 `likpf_mean` mean 0.029514、p99 0.204101。

Interpretation:

- self-GR weak boost は exp072 `likpf_mean` に対して train-side positive。
- ただし exp209 HMM/likPF blend の RMSE 10.269696 には届かない。
- worst-well regression が大きく、raw-test regeneration / inference / submit には進めない。
- 後続で使う場合は直接候補や replacement ではなく、ML / selector 側の confidence feature または regression guard 付き診断材料に限定する。
