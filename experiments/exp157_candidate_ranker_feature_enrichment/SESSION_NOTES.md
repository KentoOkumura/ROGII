# exp157_candidate_ranker_feature_enrichment セッションノート

## 2026-06-28 実装

- ユーザー依頼により `candidate_ranker_feature_enrichment` を実装開始。
- `.steering/20260628-exp157-candidate-ranker-feature-enrichment/` を作成。
- `experiments/exp157_candidate_ranker_feature_enrichment/` を `exp101_pf_candidate_ranker_or_nway_classifier` から作成。
- exp099 v2 cache に `tvt_dense*` columns が無いことを確認したため、exp072 full replay feature cache を補助 source として join する設計にした。
- 追加 candidate は `tvt_dense`、`tvt_densew`、`tvt_dense50`。`last_known_tvt + tvt_dense*_d` で絶対 TVT candidate を復元する。
- 追加 feature は dense drift、dense family dispersion、PF/Beam/likPF-vs-dense 差、tail / near-row flag、high-disagreement proxy に限定する。

## Kaggle train push 前コスト確認

- Runtime: CPU (`enable_gpu=false`)
- active selector experiment 数: 1
- LightGBM model family 数: 3
- fold 数: 5
- 合計 booster 数: 15
- exp101 control 再学習: なし
- direct TVT regression: なし
- submission / inference port: なし

## 次アクション

- py_compile、notebook JSON check、validate-exp を通す。
- Kaggle package を prepare する場合は、kernel sources に exp099 と exp072 が入っていることを確認する。

## 静的検証

- `python3 -m py_compile experiments/exp157_candidate_ranker_feature_enrichment/candidate_ranker_feature_enrichment.py experiments/exp157_candidate_ranker_feature_enrichment/settings.py`: PASS
- `python3 -m json.tool experiments/exp157_candidate_ranker_feature_enrichment/exp157_candidate_ranker_feature_enrichment_train.ipynb`: PASS
- `python3 -m json.tool experiments/exp157_candidate_ranker_feature_enrichment/exp157_candidate_ranker_feature_enrichment_inference.ipynb`: PASS
- `uv run python scripts/validate_experiment.py --experiment exp157_candidate_ranker_feature_enrichment`: PASS
- `uv run ruff check experiments/exp157_candidate_ranker_feature_enrichment/candidate_ranker_feature_enrichment.py experiments/exp157_candidate_ranker_feature_enrichment/settings.py`: PASS
- `uv run ruff format --check experiments/exp157_candidate_ranker_feature_enrichment/candidate_ranker_feature_enrichment.py experiments/exp157_candidate_ranker_feature_enrichment/settings.py`: PASS
- synthetic frame smoke for `add_feature_enrichment()`: PASS。23 generated features、97 selected numeric features、8 candidates。

## Kaggle package prepare

```bash
uv run python scripts/prepare_kaggle_notebooks.py \
  --experiment exp157_candidate_ranker_feature_enrichment \
  --notebook train \
  --kernel-id kentookumura/exp157-cand-ranker-enrich-train \
  --title 'exp157 cand ranker enrich train' \
  --run-on-push \
  --strict
```

- output: `experiments/exp157_candidate_ranker_feature_enrichment/kaggle/train`
- metadata `enable_gpu`: false
- kernel sources:
  - `kentookumura/exp099-pf-multiobs-likelihood-train`
  - `kentookumura/exp072-exp063-full-replay-feature-cache-train`

## 2026-06-29 Kaggle train 実行

```bash
kaggle kernels push -p experiments/exp157_candidate_ranker_feature_enrichment/kaggle/train
```

- result: PASS
- kernel version: 1
- URL: `https://www.kaggle.com/code/kentookumura/exp157-cand-ranker-enrich-train`

```bash
kaggle kernels pull kentookumura/exp157-cand-ranker-enrich-train \
  -p /tmp/kaggle-pull/exp157-cand-ranker-enrich-train -m
```

- result: PASS。Kaggle 側の notebook / metadata 存在確認済み。

```bash
kaggle kernels logs kentookumura/exp157-cand-ranker-enrich-train
```

- result: PASS。ただし push 直後のため通常 logs はまだ空。

```bash
timeout 300 kaggle kernels logs -f --interval 20 \
  kentookumura/exp157-cand-ranker-enrich-train
```

- result: timeout。5 分間 follow したが CLI log output は空。

```bash
kaggle kernels status kentookumura/exp157-cand-ranker-enrich-train
```

- result: `KernelWorkerStatus.RUNNING`
- 2026-06-28 21:39、21:41、21:43、21:45 UTC の status polling でも `RUNNING`。
- ユーザー指示によりローカル監視ループは停止。Kaggle notebook 実行自体は継続中。

## 2026-06-29 Kaggle train 完了確認

```bash
kaggle kernels status kentookumura/exp157-cand-ranker-enrich-train
kaggle kernels logs kentookumura/exp157-cand-ranker-enrich-train
```

- status: `KernelWorkerStatus.COMPLETE`
- logs から train 完了を確認。
- rows / wells: 3,783,989 / 773
- runtime: 10,421.758 sec
- feature count: 97
- generated dense enrichment features: 23
- model manifest SHA: `ab25fbfc0c8b92915bfbd11e62c8ffa6d84eadb3d8abf10e039927e2df7d4fb1`

```bash
kaggle kernels output kentookumura/exp157-cand-ranker-enrich-train \
  -p experiments/exp157_candidate_ranker_feature_enrichment/kaggle/output/train_v1
```

- result: PASS
- output: `experiments/exp157_candidate_ranker_feature_enrichment/kaggle/output/train_v1`

## Train v1 結果

- best OOF: `lgb_candidate_error_ranker`
- RMSE: 10.795799837
- MAE: 6.476996066
- within10: 0.792504946
- oracle label accuracy: 0.258688120
- delta RMSE vs `likpf_mean_single`: -0.799097835
- delta within10 vs `likpf_mean_single`: +0.019697467
- delta RMSE vs exp101 best OOF 11.600096615: -0.804296778
- oracle RMSE: 4.564605115
- best OOF PF selection rate: 37.1373965%
- best OOF dense family selection rate: 18.1829810%
- max path switch: 357.199 / 1000 rows
- worst well: `86454a6f`, RMSE 57.967200741

## 解釈

Train-side では supported。dense family を candidate set に入れた exp157 ranker は exp101 / likPF baseline を大きく改善した。一方で row-wise switch はまだ大きいため、direct inference port / submit はしない。次は exp157 score surface を使った segment / Viterbi continuity selector または confidence-gated fallback を確認する。
