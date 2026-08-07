# exp065_typewell_supertype_cluster_cv_audit セッションノート

## 目的

共通 typewell 候補を見つける。CV 補助監査、候補採用、提出判断は目的にしない。

## 現在の状態

- Route: pf_beam
- 状態: completed
- CV: なし
- LB: なし

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 実行済み

```bash
uv run python -m py_compile experiments/exp065_typewell_supertype_cluster_cv_audit/typewell_supertype_discovery.py experiments/exp065_typewell_supertype_cluster_cv_audit/settings.py
uv run ruff check experiments/exp065_typewell_supertype_cluster_cv_audit/typewell_supertype_discovery.py experiments/exp065_typewell_supertype_cluster_cv_audit/settings.py
uv run python -m json.tool experiments/exp065_typewell_supertype_cluster_cv_audit/exp065_typewell_supertype_cluster_cv_audit_train.ipynb
uv run python -m json.tool experiments/exp065_typewell_supertype_cluster_cv_audit/exp065_typewell_supertype_cluster_cv_audit_inference.ipynb
uv run python experiments/exp065_typewell_supertype_cluster_cv_audit/typewell_supertype_discovery.py --max-wells 40
uv run python experiments/exp065_typewell_supertype_cluster_cv_audit/typewell_supertype_discovery.py
uv run python scripts/validate_experiment.py --experiment exp065_typewell_supertype_cluster_cv_audit
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp065_typewell_supertype_cluster_cv_audit --notebook train --run-on-push --strict
uv run python scripts/record_experiment.py --experiment exp065_typewell_supertype_cluster_cv_audit --status completed --metric common_typewell_group_count --key-idea "Discover common typewell candidate groups from train typewell CSV curves with exact hash, shifted NCC, and constrained DTW; no CV or submission." --notes "Full local run completed: exact hash 752 groups / 34 duplicate wells; shifted NCC 0.98 gives 103 multi-well groups covering 314 wells; DTW 0.94 gives 63 multi-well groups covering 167 wells; 57 groups only appears under over-chained loose thresholds, so do not force it."
kaggle kernels push -p experiments/exp065_typewell_supertype_cluster_cv_audit/kaggle/train
kaggle kernels pull kentookumura/exp065-typewell-supertype-cluster-cv-audit-train -p /tmp/kaggle-pull/exp065-typewell-supertype-cluster-cv-audit-train-v1 -m
kaggle kernels logs kentookumura/exp065-typewell-supertype-cluster-cv-audit-train
timeout 180 kaggle kernels logs -f --interval 10 kentookumura/exp065-typewell-supertype-cluster-cv-audit-train
kaggle kernels output kentookumura/exp065-typewell-supertype-cluster-cv-audit-train -p /tmp/kaggle-output/exp065_typewell_supertype_cluster_cv_audit/train_v1
uv run python scripts/record_experiment.py --experiment exp065_typewell_supertype_cluster_cv_audit --status completed --metric native_overlap_group_count --key-idea "Discover common typewell groups using native typewell row-lag overlap plus exact hash, shifted NCC, and constrained DTW; Kaggle train v1 completed." --notes "Kaggle train v1 completed. Native row-lag overlap finds 10713 candidate pairs and 10697 exact containment pairs; exact native overlap clusters into 54 groups / 41 multi-well groups / 760 wells / max group 71. Example 028d7b28 vs 0dd99dc5: lag 218 rows = 109 ft, 1774 rows exact GR match, left_contained_in_right. This reproduces the shifted/trimmed typewell hypothesis better than resampled NCC/DTW."
```

## 変更点

- `typewell_supertype_discovery.py` を追加。
  - train `*__typewell.csv` の byte hash exact group を作る。
  - TVT 正規化 GR signature を作り、shifted NCC で類似ペアを抽出する。
  - NCC 候補に constrained DTW をかけ、DTW 類似ペアを抽出する。
  - native `typewell.csv` の GR 列を row lag で照合し、shift / trim で一致する pair を抽出する。
  - exact / shifted NCC / DTW の閾値別 connected components を common typewell candidate group として保存する。
- train notebook は discovery 実行と生成物確認に差し替え。
- inference notebook は提出を作らず `NO_SUBMISSION.txt` のみ作る。

## 結果

- Kaggle train v1 completed。
- output: `/tmp/kaggle-output/exp065_typewell_supertype_cluster_cv_audit/train_v1`
- wells: 773
- valid signatures: 773
- exact hash: 752 unique groups、13 multi-well groups、34 duplicate wells、max group size 10。
- native row-lag overlap:
  - pair rows: 10,713
  - exact containment pair rows: 10,697
  - exact overlap clusters: 54 unique groups、41 multi-well groups、760 wells、max group size 71。
  - `028d7b28` vs `0dd99dc5`: `row_lag_b_minus_a=218`、`row_lag_ft_equivalent=109.0`、1774 overlap rows、exact match rate 1.0、`left_contained_in_right`。
- shifted NCC:
  - `>=0.98`: 562 unique groups、103 multi-well groups、314 wells、max group size 15。
  - `>=0.99`: 698 unique groups、58 multi-well groups、133 wells、max group size 10。
  - `>=0.995`: 720 unique groups、41 multi-well groups、94 wells、max group size 10。
- constrained DTW:
  - `>=0.94`: 669 unique groups、63 multi-well groups、167 wells、max group size 10。
  - `>=0.96`: 731 unique groups、33 multi-well groups、75 wells、max group size 10。
  - `>=0.98`: 751 unique groups、14 multi-well groups、36 wells、max group size 10。
- 57 groups 付近は native row-lag overlap で 54 groups として近く再現できた。resampled NCC/DTW よりも、投稿の shifted / trimmed typewell 仮説に合う。

## 生成物

- `artifacts/typewell_well_index.csv`
- `artifacts/typewell_shifted_ncc_pairs.csv`
- `artifacts/typewell_dtw_pairs.csv`
- `artifacts/typewell_native_overlap_pairs.csv`
- `artifacts/common_typewell_cluster_assignments.csv`
- `artifacts/common_typewell_cluster_summary.csv`
- `artifacts/common_typewell_cluster_metrics.csv`
- `features/typewell_gr_signatures.npy`

## 次のアクション

1. 後続の `typewell_neighbor_prior_features` では、exact hash、native row-lag overlap、`shifted_ncc >= 0.98`、`dtw_similarity >= 0.94` を別々の group 定義として fold-safe neighbor pool に使う。
2. native overlap group は大きくなりやすいため、同一 group 内でも TVT offset / row lag / overlap fraction を条件に使う。
