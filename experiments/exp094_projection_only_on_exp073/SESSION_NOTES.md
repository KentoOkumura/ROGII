# exp094_projection_only_on_exp073 セッションノート

## 目的

`projection_only_on_exp073` backlog を実装する。exp073 の fold-out OOF prediction を固定し、`TVT + Z - prefix_anchor` 空間の robust polynomial projection だけを後処理として比較する。

## 現在の状態

- Route: ml_model
- 状態: completed_no_inference_guard_failed
- CV: 9.399456024
- LB: まだなし
- inference: `inference.selected_variant` が null のため無効

## コマンドログ

### 2026-06-20 実装

```bash
make new-steering EXP=exp094_projection_only_on_exp073
make new-exp EXP=exp094_projection_only_on_exp073
.venv/bin/python -m py_compile experiments/exp094_projection_only_on_exp073/projection_only_on_exp073.py
make validate-exp EXP=exp094_projection_only_on_exp073
make prepare-kaggle-notebooks EXP=exp094_projection_only_on_exp073 EXTRA_ARGS="--notebook train --run-on-push --strict"
make prepare-kaggle-notebooks EXP=exp094_projection_only_on_exp073 EXTRA_ARGS="--notebook inference --run-on-push --strict"
```

実装した内容:

- `docs/legacy/steering/20260620-exp094-projection-only-on-exp073/` を作成し、requirements/design/tasklist を記入。
- `config.yaml` に exp073 OOF prediction、projection grid、selection guard、Kaggle source を記入。
- `projection_only_on_exp073.py` を追加。
- train/inference notebook を projection audit 用に更新。
- README/result/metrics を未実行状態で更新。

確認結果:

- `make validate-exp` は strict で PASS。
- Kaggle train package は `experiments/exp094_projection_only_on_exp073/kaggle/train` に生成済み。
- `kernel-metadata.json` は `enable_gpu=false`、`enable_internet=false`、`kernel_sources=["kentookumura/exp073-full-replay-repro-guard-train"]`。
- 未選択状態の inference package も生成し、metadata は `enable_gpu=false`、`enable_internet=false`、`kernel_sources=["kentookumura/exp073-full-replay-repro-guard-infer"]`。
- debug smoke として `/tmp/exp094_projection_smoke` に `selected_model=lgb0` / `max_rows=5000` で補助スクリプトを直接実行し、コード経路が完走することを確認した。これは公式評価ではない。
- Kaggle runtime で raw data path が `data/raw/...` にならないよう、notebook 側で `ExperimentPaths` の解決済み `train_dir/test_dir/sample_submission` を config に反映するよう修正し、train package を再生成した。

### 未実行

- inference port は guard failed のため実行しない。

### 2026-06-21 Kaggle train v1

最初の push は title slug mismatch で Kaggle API 400 になった。

```bash
make prepare-kaggle-notebooks EXP=exp094_projection_only_on_exp073 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp094-projection-only-on-exp073-train --title 'exp094 projection only on exp073 train' --run-on-push --strict"
make push-kaggle-train EXP=exp094_projection_only_on_exp073
kaggle kernels pull kentookumura/exp094-projection-only-on-exp073-train -p /tmp/kaggle-pull/exp094-projection-only-on-exp073-train-v1 -m
kaggle kernels logs kentookumura/exp094-projection-only-on-exp073-train
kaggle kernels output kentookumura/exp094-projection-only-on-exp073-train -p /tmp/kaggle-output/exp094_projection_only_on_exp073/train_v1
```

結果:

- Kernel: `kentookumura/exp094-projection-only-on-exp073-train` v1
- URL: https://www.kaggle.com/code/kentookumura/exp094-projection-only-on-exp073-train
- Output: `/tmp/kaggle-output/exp094_projection_only_on_exp073/train_v1`
- Status: COMPLETED
- Runtime: 3412.934 秒
- Rows / wells: 3,783,989 / 773
- Baseline exp073 OOF RMSE: 9.526374817
- Best variant: `degree4_beta0.75_c2`
- Best RMSE: 9.399456024
- Delta vs baseline: -0.126918725
- Best prediction SHA: `bc4f02808ae1fd1cc0a174ee558cfee462734961ae51ef9a65a1018c38889200`
- Input exp073 decompressed content SHA: `fd6c68050058c40b4960f3ff2af9905bfcb1c12d540c71e331d0aa85ca9756a4`
- Raw prefix anchor check: max abs diff vs exp073 `last_known_tvt` = 0.0

Guard:

- `passes_guard=false`
- `recommendation=do_not_port_without_review`
- max fold regression: -0.090495110
- near row regression, distance 0-50 ft: +1.439466
- short tail regression: -0.049640
- 全 variant の guard 通過数: 0

結論:

- Overall OOF は大きく改善したが、prefix 直後を壊すため fixed projection-only policy は inference port しない。
- projection を続けるなら near-prefix を除外する long-tail-only / confidence-gated projection として別実験に切る。

## 変更点

- exp073 `gpu_repro_guard_dp_threads8` / `lgb_mean` OOF prediction を chunk read し、selected model のみ評価する。
- raw train well file から `MD/Z/TVT_input` を復元し、`last_known_tvt` と prefix anchor の一致を max abs 0.05 ft 以内で検証する。
- degree 3/4/5、beta 0.25/0.50/0.75、robust C 1.25/1.5/2.0 を比較する。
- variant metrics、fold metrics、bucket metrics、by-well metrics、best predictions、summary JSON を保存する。

## 再現性メモ

- seed policy: `no_new_rng_projection_postprocess`
- stochastic components: upstream exp073 GPU LightGBM OOF prediction のみ。exp094 自体に RNG はない。
- CPU/GPU runtime: CPU で十分。LightGBM/GPU 学習なし。
- Kaggle kernel id / version: `kentookumura/exp094-projection-only-on-exp073-train` v1
- input / feature schema SHA: exp073 prediction raw file SHA `986e26c5c6617ade714623d44433e9beacdb2b1027d46c4a4e70825bc8ab87fc`、decompressed content SHA `fd6c68050058c40b4960f3ff2af9905bfcb1c12d540c71e331d0aa85ca9756a4`
- feature content SHA: projection context は raw train から再構成。gzip prediction は decompressed content SHA を主証拠にする。
- model manifest / model SHA: 対象外
- prediction SHA: best prediction SHA `bc4f02808ae1fd1cc0a174ee558cfee462734961ae51ef9a65a1018c38889200`
- submission SHA: inference 未選択のため未記録
- rerun check: 未実施

## 次のアクション

1. exp094 は完了。inference port は行わない。
2. projection を続ける場合は `projection_confidence_error_map` または near-prefix excluded long-tail gate として別実験に切る。
