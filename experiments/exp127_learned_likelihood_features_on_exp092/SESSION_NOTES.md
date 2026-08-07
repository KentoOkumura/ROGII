# exp127_learned_likelihood_features_on_exp092 セッションノート

## 状態

- 2026-06-25: 実装済み。GPU quota 上限で Kaggle train は未実行。
- 2026-06-27: Kaggle train v1 完了。shared-row control に対して learned likelihood add-only feature は全 pooled model で改善。提出なし。
- Route: `ml_model`
- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- feature 親: `exp112_learned_pf_likelihood_weight_or_feature_followup`

## 実装メモ

- `learned_likelihood_gate_rawtest_parity_or_ml_feature` の残作業として、分岐Bの exp092 add-only feature 評価を `exp127` に切り出した。
- exp092 の U-projection correction plus disagreement surface を維持する。
- exp112 `ml_features.csv.gz` は 155 wells subset のため、exp072/exp092 surface と exp112 feature cache の shared rows だけで control と add-only variant を比較する。
- Inference notebook は no-submission summary のみを書き、`submission.csv` は作らない。

## コマンド

```bash
make new-steering EXP=exp127_learned_likelihood_features_on_exp092
make new-exp EXP=exp127_learned_likelihood_features_on_exp092 SOURCE=experiments/exp092_u_projection_correction_disagreement_fullrun
make prepare-kaggle-notebooks EXP=exp127_learned_likelihood_features_on_exp092 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp127-ll-features-exp092-train --title 'exp127 ll features exp092 train' --run-on-push --strict"
make push-kaggle-train EXP=exp127_learned_likelihood_features_on_exp092
make prepare-kaggle-notebooks EXP=exp127_learned_likelihood_features_on_exp092 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp127-train --title 'exp127 train' --run-on-push --strict"
make push-kaggle-train EXP=exp127_learned_likelihood_features_on_exp092
kaggle kernels status kentookumura/exp127-train
kaggle kernels output kentookumura/exp127-train -p experiments/exp127_learned_likelihood_features_on_exp092/kaggle/output/train_v1
```

## 次アクション

- direct inference port / submit はしない。
- exp127 の feature family を使う場合は、exp115 hidden-like stress、raw-test/full-train feature parity、worst-well guard を先に通す。
- `segment_level_dense_candidate_verifier` や他の exp092 confidence feature 実験では、exp127 learned likelihood confidence を候補 feature として参照する。

## 検証

- `.venv/bin/python -m py_compile experiments/exp127_learned_likelihood_features_on_exp092/learned_likelihood_features_on_exp092.py experiments/exp127_learned_likelihood_features_on_exp092/settings.py`: PASS
- `python3 -m json.tool experiments/exp127_learned_likelihood_features_on_exp092/exp127_learned_likelihood_features_on_exp092_train.ipynb`: PASS
- `python3 -m json.tool experiments/exp127_learned_likelihood_features_on_exp092/exp127_learned_likelihood_features_on_exp092_inference.ipynb`: PASS
- `.venv/bin/ruff check experiments/exp127_learned_likelihood_features_on_exp092/learned_likelihood_features_on_exp092.py experiments/exp127_learned_likelihood_features_on_exp092/settings.py`: PASS
- `make validate-exp EXP=exp127_learned_likelihood_features_on_exp092`: PASS
- `make prepare-kaggle-notebooks EXP=exp127_learned_likelihood_features_on_exp092 EXTRA_ARGS="--notebook train --run-on-push --strict"`: PASS
- `make prepare-kaggle-notebooks EXP=exp127_learned_likelihood_features_on_exp092 EXTRA_ARGS="--notebook inference --run-on-push --strict"`: PASS
- `make update-summary`: PASS、`experiment_summary.md` に exp127 を追加。

## Kaggle train push

- 初回 package は default title `ROGII - Wellbore Geology Prediction exp127_learned_likelihood_features_on_exp092 train` と id `kentookumura/exp127-learned-likelihood-features-on-exp092-train` の slug が一致せず、`SaveKernel` 400 で失敗した。
- 短い canonical kernel id / title に変更して再生成した。
  - kernel id: `kentookumura/exp127-ll-features-exp092-train`
  - title: `exp127 ll features exp092 train`
- 再 push は Kaggle API には到達したが、アカウントの weekly GPU quota 上限で実行拒否された。
  - message: `Maximum weekly GPU quota of 45.00 hours reached.`
  - status: Kaggle train 未実行
- GPU quota 回復後、`kentookumura/exp127-ll-features-exp092-train` への push は Kaggle 側で `Notebook not found` になった。Kaggle list でも exp127 の該当 notebook が見えなかったため、slug/title 一致の canonical id に再生成した。
  - kernel id: `kentookumura/exp127-train`
  - title: `exp127 train`
- `make push-kaggle-train EXP=exp127_learned_likelihood_features_on_exp092`: PASS
  - `Kernel version 1 successfully pushed.`
  - URL: https://www.kaggle.com/code/kentookumura/exp127-train
- `kaggle kernels status kentookumura/exp127-train`: `KernelWorkerStatus.COMPLETE`
- `kaggle kernels output kentookumura/exp127-train -p experiments/exp127_learned_likelihood_features_on_exp092/kaggle/output/train_v1`: PASS

## Kaggle train v1 結果

- 評価範囲: 757,738 rows / 155 wells。exp072/exp092 base surface は 3,783,989 rows / 773 wells だが、exp112 feature cache の存在する shared rows に限定した。
- Runtime: 7133.981 sec
- Active mode: `gpu_repro_guard_dp_threads8`
- Model count: 30
- Output: `experiments/exp127_learned_likelihood_features_on_exp092/kaggle/output/train_v1`

| variant | model | features | RMSE |
| --- | --- | ---: | ---: |
| `exp092_shared_row_control` | `lgb0` | 240 | 10.022150359 |
| `exp092_shared_row_control` | `lgb1` | 240 | 9.865476965 |
| `exp092_shared_row_control` | `lgb2` | 240 | 9.872184867 |
| `exp092_shared_row_control` | `lgb_mean` | 240 | 9.847052694 |
| `learned_likelihood_confidence_addonly` | `lgb0` | 294 | 9.867141369 |
| `learned_likelihood_confidence_addonly` | `lgb1` | 294 | 9.773581370 |
| `learned_likelihood_confidence_addonly` | `lgb2` | 294 | 9.753643527 |
| `learned_likelihood_confidence_addonly` | `lgb_mean` | 294 | 9.727317518 |

Delta add-only minus control:

- `lgb0`: -0.155008989
- `lgb1`: -0.091895595
- `lgb2`: -0.118541340
- `lgb_mean`: -0.119735177

Bucket は `000_050`、`050_100`、`100_250`、`250_500`、`500_1000`、`1000_plus` の全てで `lgb_mean` が改善した。by-well `lgb_mean` delta は mean -0.042303、median -0.048879、p75 +0.199447、max regression +1.071012、best improvement -2.155674。

Top learned feature importance:

- `ll_candidate_tvt_likpf_mean_minus_last_known_tvt`
- `ll_learned_prob_beam_mean`
- `ll_learned_pred_abs_error_beam_mean`
- `ll_learned_prob_weighted_tvt_minus_last_known_tvt`
- `ll_candidate_tvt_beam_mean_minus_last_known_tvt`

主要 SHA:

- metrics CSV: `6e05bc6bff630a64fdf409df17e76fc5332abaf7bf4ec588ea7268feaf6e7f20`
- predictions gzip: `6f1f5f36214255019d815af33d19360f13b916a82f5331436cd8dd6a5d48a55e`
- predictions decompressed: `7c11f3ee25c21c7b92f86c65773521f327ad66d98577c9da3b75b9c00d1508a2`
- exp112 feature decompressed source: `56c0f62238abfc89f05e5700341344c15815bd3a5f93e5b0a6a079a661b9411e`

結論: shared rows 上では learned likelihood confidence add-only は支持。subset 限定かつ worst-well regression が残るため、提出候補にはせず、hidden-like stress / raw-test parity を通してから後続の confidence feature または segment verifier へ渡す。
