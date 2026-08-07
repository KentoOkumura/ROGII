# exp154_segment_level_dense_candidate_verifier_on_exp148 セッションノート

## 現在の状態

- status: `submitted_public_lb_8p078_not_adopted`
- route: `ml_model`
- parent: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- baseline: exp148 `lgb_mean` CV 8.501281182 / Public LB 7.960
- LightGBM training: なし
- inference: Kaggle inference v1 complete, submission generated and submitted
- submission: ref `54142393`, Public LB 8.078
- decision: exp148 Public LB 7.960 より +0.118 悪化したため、ML route anchor には採用しない。

## 実装内容

- `.steering/20260628-exp154-segment-level-dense-candidate-verifier-on-exp148/` を作成。
- `experiments/exp154_segment_level_dense_candidate_verifier_on_exp148/` を exp135 からコピーして作成。
- backlog の比較対象を旧 exp092 から現 ML route submitted anchor の exp148 に更新した。
- 実装本体を `segment_level_dense_candidate_verifier_on_exp148.py` にリネームし、base prediction を exp148 `lgb_mean` OOF に変更した。
- exp072 PF/Beam/dense feature cache と exp073 reference OOF は引き続き比較・診断用に読む。
- exp135 の単純 segment gate に対して、near guard と candidate path continuity guard を追加した。
- `config.yaml` を no-new-LGBM の posthoc segment verifier audit として更新した。

## 設計メモ

- LightGBM の新規学習は行わない。
- `target_tvt` は scoring、oracle coverage readout、posthoc error 集計だけに使う。
- verifier 条件は `dense_std_abs`、`tvt_dense_d_abs`、`pf_dense_abs_diff`、`base_dense_abs_diff`、`pf_beam_abs_diff`、`tail_rank`、`md_since`、candidate path step に限定する。
- near `000_050` を壊さないため、segment 採用には `max_near_md_since` guard を入れる。
- row-wise switch ではなく、`min_segment_rows` と candidate path step guard を通った連続 segment だけを採用する。
- 合否は exp148 に対する overall RMSE だけでなく、PF worst50、common PF+ML worst26、near-row、path switch、worst-well regression、raw-test feature parity で見る。

## 実行予定

```bash
uv run python -m py_compile experiments/exp154_segment_level_dense_candidate_verifier_on_exp148/segment_level_dense_candidate_verifier_on_exp148.py experiments/exp154_segment_level_dense_candidate_verifier_on_exp148/settings.py
uv run ruff check experiments/exp154_segment_level_dense_candidate_verifier_on_exp148/segment_level_dense_candidate_verifier_on_exp148.py experiments/exp154_segment_level_dense_candidate_verifier_on_exp148/settings.py
uv run ruff format --check experiments/exp154_segment_level_dense_candidate_verifier_on_exp148/segment_level_dense_candidate_verifier_on_exp148.py experiments/exp154_segment_level_dense_candidate_verifier_on_exp148/settings.py
make validate-exp EXP=exp154_segment_level_dense_candidate_verifier_on_exp148
```

Kaggle train push 前の booster count:

- active variant 数: 0 train variants
- LightGBM config 数: 0
- folds: 0
- total boosters: 0
- control retraining: なし

Kaggle で audit を走らせる場合:

```bash
make prepare-kaggle-notebooks EXP=exp154_segment_level_dense_candidate_verifier_on_exp148 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp154-segment-dense-verifier-train --title 'exp154 segment dense verifier train' --run-on-push --strict"
make push-kaggle-train EXP=exp154_segment_level_dense_candidate_verifier_on_exp148
```

## Kaggle train v1

```bash
make prepare-kaggle-notebooks EXP=exp154_segment_level_dense_candidate_verifier_on_exp148 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp154-segment-dense-verifier-train --title 'exp154 segment dense verifier train' --run-on-push --strict"
make push-kaggle-train EXP=exp154_segment_level_dense_candidate_verifier_on_exp148
kaggle kernels status kentookumura/exp154-segment-dense-verifier-train
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp154-segment-dense-verifier-train
```

- kernel: `kentookumura/exp154-segment-dense-verifier-train`
- version: 1
- push: success
- URL: `https://www.kaggle.com/code/kentookumura/exp154-segment-dense-verifier-train`
- status after push: `KernelWorkerStatus.RUNNING`
- live logs: initially empty via CLI
- monitoring: stopped per user request; user will report completion.

## Kaggle train v1 completion

```bash
kaggle kernels status kentookumura/exp154-segment-dense-verifier-train
kaggle kernels logs kentookumura/exp154-segment-dense-verifier-train
kaggle kernels output kentookumura/exp154-segment-dense-verifier-train -p experiments/exp154_segment_level_dense_candidate_verifier_on_exp148/kaggle/output/train_v1
```

- status: `KernelWorkerStatus.COMPLETE`
- output: `experiments/exp154_segment_level_dense_candidate_verifier_on_exp148/kaggle/output/train_v1`
- rows / wells: 3,783,989 / 773
- runtime in summary: 199.289 sec
- base `exp148_lgb_mean`: RMSE 8.501281182 / MAE 5.335650921 / within10 0.856332035
- best non-oracle: `verifier_dense50_tail1500_q90_min80_clip10_a025`
- best RMSE: 8.472279555
- delta vs exp148: -0.029001627
- best within10: 0.855105023, slightly worse than base by -0.001227012
- gate rate / wells: 0.052558 / 165 wells
- max well regression: +2.287373
- improved / worsened wells: 75 / 90
- PF `likpf_mean` worst50: delta -0.480663
- common PF+ML worst26: delta -0.618472
- exp148 worst50: delta -0.292035
- near `000_050`: delta 0.000000
- `1000_plus`: delta -0.035276
- `1000_plus + pf_dense_diff_q4`: delta -0.089011
- mid buckets worsened: `250_500` +0.014269, `500_1000` +0.026098
- raw-test parity checklist: required columns present, gate conditions target-free, no LightGBM training.
- decision at train completion: train-side supported, but within10/worst-well risk remained. User requested direct submit because submission slots were available.

## Kaggle inference v1 / submit

```bash
make prepare-kaggle-notebooks EXP=exp154_segment_level_dense_candidate_verifier_on_exp148 EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp154-segment-dense-verifier-inference --title 'exp154 segment dense verifier inference' --run-on-push --strict"
make push-kaggle-infer EXP=exp154_segment_level_dense_candidate_verifier_on_exp148
kaggle kernels status kentookumura/exp154-segment-dense-verifier-inference
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp154-segment-dense-verifier-inference
kaggle kernels output kentookumura/exp154-segment-dense-verifier-inference -p experiments/exp154_segment_level_dense_candidate_verifier_on_exp148/kaggle/output/inference_v1
python3 .agents/skills/kaggle-submit-check/scripts/check_submission.py experiments/exp154_segment_level_dense_candidate_verifier_on_exp148/kaggle/output/inference_v1/submission.csv --sample data/raw/sample_submission.csv
kaggle competitions submit rogii-wellbore-geology-prediction -k kentookumura/exp154-segment-dense-verifier-inference -v 1 -f submission.csv -m "exp154 segment dense verifier on exp148"
```

- kernel: `kentookumura/exp154-segment-dense-verifier-inference`
- version: 1
- status: `KernelWorkerStatus.COMPLETE`
- URL: `https://www.kaggle.com/code/kentookumura/exp154-segment-dense-verifier-inference`
- output: `experiments/exp154_segment_level_dense_candidate_verifier_on_exp148/kaggle/output/inference_v1`
- selected verifier: `verifier_dense50_tail1500_q90_min80_clip10_a025`
- submission rows / wells: 14,151 / 3
- fallback rows: 0
- changed rows vs exp148: 951 (`0.067203731`)
- changed wells vs exp148: 1
- abs delta vs exp148: mean 0.168009326 / p95 2.5 / max 2.5
- prediction range: 11590.202148 - 12240.267578
- submission SHA256: `fb6a2be0eb9082974f23806690ffea7552215717b1f6a83d406d6f0da2db1d54`
- submit-check: PASS

Kaggle submission:

- ref: `54142393`
- description: `exp154 segment dense verifier on exp148`
- submitted_at_utc: `2026-06-28 14:14:56.697000`
- status: `SubmissionStatus.COMPLETE`
- Public LB: 8.078
- Private LB: 未公開
- comparison: exp148 Public LB 7.960 に対して +0.118 悪化。
- decision: submit 済みだが採用しない。ML route submitted anchor は exp148 のまま。
