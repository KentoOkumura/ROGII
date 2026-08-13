# exp144_learned_likelihood_hidden_stress_and_rawtest_parity セッションノート

## 状態

- 2026-06-27: Kaggle train v1 完了。hidden-like stress は支持、raw-test parity は未充足。提出なし。
- Route: `ml_model`
- 親実験: `exp127_learned_likelihood_features_on_exp092`
- feature 親: `exp112_learned_pf_likelihood_weight_or_feature_followup`
- split 親: `exp115_hidden_like_spatial_holdout_from_ppt`

## 実装メモ

- ユーザー指定名は `exp127...` だったが、既存最新が `exp143` のため `exp144_learned_likelihood_hidden_stress_and_rawtest_parity` として作成した。
- exp127 の保存済み row-level OOF predictions を読み、control と add-only を hidden-like split で readout する。
- 新規学習、submission.csv 生成、inference port は行わない。
- raw-test parity checklist では、exp112 raw-test feature regeneration が未実装なら fail として記録する。

## コマンド

```bash
make new-steering EXP=exp144_learned_likelihood_hidden_stress_and_rawtest_parity
make new-exp EXP=exp144_learned_likelihood_hidden_stress_and_rawtest_parity SOURCE=experiments/exp127_learned_likelihood_features_on_exp092
.venv/bin/python -m py_compile experiments/exp144_learned_likelihood_hidden_stress_and_rawtest_parity/learned_likelihood_hidden_stress_and_rawtest_parity.py experiments/exp144_learned_likelihood_hidden_stress_and_rawtest_parity/settings.py
.venv/bin/ruff check experiments/exp144_learned_likelihood_hidden_stress_and_rawtest_parity/learned_likelihood_hidden_stress_and_rawtest_parity.py experiments/exp144_learned_likelihood_hidden_stress_and_rawtest_parity/settings.py
python3 -m json.tool experiments/exp144_learned_likelihood_hidden_stress_and_rawtest_parity/exp144_learned_likelihood_hidden_stress_and_rawtest_parity_train.ipynb
python3 -m json.tool experiments/exp144_learned_likelihood_hidden_stress_and_rawtest_parity/exp144_learned_likelihood_hidden_stress_and_rawtest_parity_inference.ipynb
make validate-exp EXP=exp144_learned_likelihood_hidden_stress_and_rawtest_parity
make prepare-kaggle-notebooks EXP=exp144_learned_likelihood_hidden_stress_and_rawtest_parity EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp144-train --title 'exp144 train' --run-on-push --strict"
make prepare-kaggle-notebooks EXP=exp144_learned_likelihood_hidden_stress_and_rawtest_parity EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp144-inference --title 'exp144 inference' --strict"
.venv/bin/python -m py_compile experiments/exp144_learned_likelihood_hidden_stress_and_rawtest_parity/kaggle/train/learned_likelihood_hidden_stress_and_rawtest_parity.py experiments/exp144_learned_likelihood_hidden_stress_and_rawtest_parity/kaggle/train/settings.py experiments/exp144_learned_likelihood_hidden_stress_and_rawtest_parity/kaggle/inference/settings.py
.venv/bin/ruff check experiments/exp144_learned_likelihood_hidden_stress_and_rawtest_parity/kaggle/train/learned_likelihood_hidden_stress_and_rawtest_parity.py experiments/exp144_learned_likelihood_hidden_stress_and_rawtest_parity/kaggle/train/settings.py
make push-kaggle-train EXP=exp144_learned_likelihood_hidden_stress_and_rawtest_parity
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp144-train
kaggle kernels logs kentookumura/exp144-train
kaggle kernels status kentookumura/exp144-train
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp144-train
kaggle kernels output kentookumura/exp144-train -p experiments/exp144_learned_likelihood_hidden_stress_and_rawtest_parity/kaggle/output/train_v1
```

## 検証

- `py_compile`: PASS
- `ruff check`: PASS
- train notebook JSON: PASS
- inference notebook JSON: PASS
- `make validate-exp`: PASS
- train package prepare: PASS (`kentookumura/exp144-train`, CPU, internet off, run-on-push)
- inference package prepare: PASS (`kentookumura/exp144-inference`, CPU, internet off, no run-on-push)
- generated package `py_compile`: PASS
- generated package `ruff check`: PASS
- `make push-kaggle-train`: PASS
- Kaggle train v1: COMPLETE
- Output取得: PASS

## Kaggle train v1 結果

- Kernel: `kentookumura/exp144-train`
- URL: https://www.kaggle.com/code/kentookumura/exp144-train
- Output: `experiments/exp144_learned_likelihood_hidden_stress_and_rawtest_parity/kaggle/output/train_v1`

Decision:

- `hidden_like_lgb_mean_supported=true`
- `rawtest_parity_all_pass=false`
- `max_hidden_like_well_regression_lgb_mean=1.070999934`
- `direct_submission_candidate=not_selected`
- recommendation: `hidden_like_supported_but_rawtest_parity_missing`

`lgb_mean` focus:

| split | rows | control RMSE | add-only RMSE | delta |
| --- | ---: | ---: | ---: | ---: |
| all_shared_rows | 757,738 | 9.847053 | 9.727318 | -0.119735 |
| verification_like_spatial | 169,691 | 13.037491 | 12.760311 | -0.277180 |
| verification_like_typewell_purged | 166,972 | 13.082838 | 12.787921 | -0.294917 |

raw-test parity checklist は pass 5 / fail 3。fail は full-train coverage 155/773 wells、exp112 raw-test feature regeneration missing、hidden submission candidate not selected。

## 次アクション

- direct inference port / submit はしない。
- exp112 learned likelihood feature の raw-test target-free generator と schema parity audit を作れるか検討する。
- `experiment_summary.md` と `backlog/KAGGLE_DIRECTION.md` を更新する。
