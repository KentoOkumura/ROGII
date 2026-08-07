# exp116_hidden_like_anchor_score_readout_on_exp115 セッションノート

## 目的

`exp115_hidden_like_spatial_holdout_from_ppt` の保存済み Kaggle output を正の hidden-like split とし、既存 anchor の OOF / train-side prediction を再学習なしで採点する。これは exp115 split で ML を学習し直す実験ではなく、通常 OOF が hidden-like well subset で崩れないかを見る stress readout。

## 現在の状態

- Route: ml_model
- 状態: local readout 完了
- CV: 新規 CV なし
- LB: なし

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

```bash
make new-steering EXP=exp116_hidden_like_anchor_score_readout_on_exp115
make new-exp EXP=exp116_hidden_like_anchor_score_readout_on_exp115
.venv/bin/python -m py_compile experiments/exp116_hidden_like_anchor_score_readout_on_exp115/hidden_like_anchor_score_readout_on_exp115.py
.venv/bin/python experiments/exp116_hidden_like_anchor_score_readout_on_exp115/hidden_like_anchor_score_readout_on_exp115.py --allow-local --max-sources 1
.venv/bin/python experiments/exp116_hidden_like_anchor_score_readout_on_exp115/hidden_like_anchor_score_readout_on_exp115.py --allow-local
.venv/bin/python -m json.tool experiments/exp116_hidden_like_anchor_score_readout_on_exp115/exp116_hidden_like_anchor_score_readout_on_exp115_train.ipynb
.venv/bin/ruff check --fix experiments/exp116_hidden_like_anchor_score_readout_on_exp115/hidden_like_anchor_score_readout_on_exp115.py
.venv/bin/ruff format experiments/exp116_hidden_like_anchor_score_readout_on_exp115/hidden_like_anchor_score_readout_on_exp115.py
.venv/bin/ruff check experiments/exp116_hidden_like_anchor_score_readout_on_exp115/hidden_like_anchor_score_readout_on_exp115.py
make validate-exp EXP=exp116_hidden_like_anchor_score_readout_on_exp115
make prepare-kaggle-notebooks EXP=exp116_hidden_like_anchor_score_readout_on_exp115 EXTRA_ARGS="--notebook train --strict"
make prepare-kaggle-notebooks EXP=exp116_hidden_like_anchor_score_readout_on_exp115 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp116-hidden-like-anchor-readout-train --title 'exp116 hidden like anchor readout train' --strict"
make prepare-kaggle-notebooks EXP=exp116_hidden_like_anchor_score_readout_on_exp115 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp116-hidden-like-anchor-readout-train --title 'exp116 hidden like anchor readout train' --run-on-push --strict"
make push-kaggle-train EXP=exp116_hidden_like_anchor_score_readout_on_exp115
kaggle kernels pull kentookumura/exp116-hidden-like-anchor-readout-train -p /tmp/kaggle-pull/exp116-hidden-like-anchor-readout-train-v1 -m
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp116-hidden-like-anchor-readout-train
make push-kaggle-train EXP=exp116_hidden_like_anchor_score_readout_on_exp115
kaggle kernels pull kentookumura/exp116-hidden-like-anchor-readout-train -p /tmp/kaggle-pull/exp116-hidden-like-anchor-readout-train-v2 -m
timeout 600 kaggle kernels logs -f --interval 30 kentookumura/exp116-hidden-like-anchor-readout-train
kaggle kernels output kentookumura/exp116-hidden-like-anchor-readout-train -p experiments/exp116_hidden_like_anchor_score_readout_on_exp115/kaggle/output/train_v2
kaggle kernels status kentookumura/exp116-hidden-like-anchor-readout-train
```

## 変更点

- `.steering/20260624-exp116-hidden-like-anchor-score-readout-on-exp115/` に要件、設計、タスクを記録。
- `config.yaml` に exp115 split artifact と exp073 / exp098 row prediction、exp092 by-well metrics の入力候補を追加。
- `hidden_like_anchor_score_readout_on_exp115.py` を追加。row prediction は gzip を chunk 読み込みし、exp115 valid wells だけを保持して集計する。
- train notebook を、入力 split / source inventory / readout 実行 / metrics preview が見える構成に更新。
- local readout output を `artifacts/` と `metrics.json` に保存。

## local readout 結果

- loaded sources: 3。missing sources: 0。
- `exp073` / `exp098`: row-level prediction から scoring。
- `exp092`: row-level prediction が手元の output で空だったため、`/tmp/exp092_train_output_check/artifacts/exp092_u_projection_correction_disagreement_fullrun_by_well.csv` の by-well metrics から weighted RMSE を集計。eval-rank bucket のような row-level bucket は exp092 では不可。

`verification_like_spatial` overall:

| source | model | input | RMSE | rows | wells |
| --- | --- | --- | ---: | ---: | ---: |
| exp073 | lgb2 | row | 10.765221 | 972463 | 200 |
| exp098 | lgb2 | row | 10.795376 | 972463 | 200 |
| exp073 | lgb1 | row | 10.802345 | 972463 | 200 |
| exp073 | lgb_mean | row | 10.806643 | 972463 | 200 |
| exp092 | lgb1 | by-well | 10.832060 | 972463 | 200 |

`verification_like_typewell_purged` overall:

| source | model | input | RMSE | rows | wells |
| --- | --- | --- | ---: | ---: | ---: |
| exp073 | lgb2 | row | 10.725383 | 976449 | 200 |
| exp098 | lgb2 | row | 10.750165 | 976449 | 200 |
| exp073 | lgb1 | row | 10.756291 | 976449 | 200 |
| exp098 | lgb1 | row | 10.766579 | 976449 | 200 |
| exp073 | lgb_mean | row | 10.769013 | 976449 | 200 |

## 再現性メモ

- seed policy: 新規乱数なし。CSV deterministic merge / groupby のみ。
- stochastic components: 新規なし。upstream の exp073 PF/Beam / LightGBM、exp092 LightGBM、exp098 LightGBM は参照 prediction として扱う。
- CPU/GPU runtime: CPU only。GPU 不要。
- Kaggle kernel id / version: package id `kentookumura/exp116-hidden-like-anchor-readout-train`。未 push。
- input SHA:
  - exp073 prediction decompressed SHA256: `fd6c68050058c40b4960f3ff2af9905bfcb1c12d540c71e331d0aa85ca9756a4`
  - exp098 prediction decompressed SHA256: `3780d7a158276ae9c8025463728758c61f7bdc529d57eab653ff0b80431849d8`
  - exp092 by-well SHA256: `6cc40f7f766f98efc6556cd7a7cfd990c846c484ba74e287bda0682206b03c7e`
- feature content SHA: 新規 feature cache なし。readout CSV SHA は `metrics.json` の `readout.artifact_sha256` に保存。
- model manifest / model SHA: 新規モデルなし。
- prediction SHA: 新規 prediction なし。upstream prediction SHA は source inventory に保存。
- submission SHA: なし。
- rerun check: `--max-sources 1` smoke と full local readout が完了。
- package check: train package generated at `experiments/exp116_hidden_like_anchor_score_readout_on_exp115/kaggle/train/` with GPU off, internet off, and upstream kernel sources for exp115 / exp073 / exp098 / exp092.

## Kaggle train v1

- Kernel: `kentookumura/exp116-hidden-like-anchor-readout-train`
- Version: v1
- URL: `https://www.kaggle.com/code/kentookumura/exp116-hidden-like-anchor-readout-train`
- Result: failed before scoring.
- Failure reason: fixed `/kaggle/input/<kernel-slug>/artifacts/...` path candidates did not match the mounted upstream output directory names, so notebook inventory saw exp115 split paths as `None` and failed at `pd.read_csv(None)`.
- Fix: `first_existing_path()` now falls back to searching `/kaggle/input/**/<basename>` when running on Kaggle. This avoids hard-coding the mounted kernel source directory name.

## Kaggle train v2

- Kernel: `kentookumura/exp116-hidden-like-anchor-readout-train`
- Version: v2
- Status: `KernelWorkerStatus.COMPLETE`
- Output: `experiments/exp116_hidden_like_anchor_score_readout_on_exp115/kaggle/output/train_v2/`
- Runtime evidence: log reached output write; artifact files exist under `/kaggle/working/artifacts`.
- loaded sources: 3。missing sources: 0。
- exp115 split mount resolved to `/kaggle/input/notebooks/kentookumura/exp115-hidden-like-spatial-holdout-from-ppt-train/artifacts/...` via basename search.
- Kaggle artifact SHA:
  - `overall_metrics`: `610384ff1d5e22980a631e59459fda8961722dab7d0fc4a3c7fda425a8b0b321`
  - `bucket_metrics`: `aba6e1520c08b1771a9749267ead55c93affa9be045d6e1dc634672226b56904`
  - `by_well`: `5fcfd6ba66048613beb047a77d8e10a465cba3781f2337358d3993433739509c`
  - `worst_well_delta`: `e5d569532a61b943ae0d67d7995a2c96ead8503e0b6a4deed783161d439041b5`
  - `source_inventory`: `43ccbf1e2c6c41394d7c5738fac55f6cfe293de5d2d9314ab33c8361c266910f`
  - `summary`: `32e92d65b764b6df968bbd2ae8de96a7acd01a72452dee6abcc689ec91ee7800`

Kaggle v2 の score は local readout と同じ主要値。source inventory / summary SHA は Kaggle mount path を含むため local run と異なる。

## 次のアクション

1. exp116 は診断として閉じる。exp092 の row-level prediction が正式に取得できた場合だけ、同じ script で exp092 row bucket を追加再実行する。
