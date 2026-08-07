# exp145_learned_likelihood_rawtest_feature_generator_parity セッションノート

## 目的

exp112 learned likelihood features を full train / raw test で target-free に再生成する generator と schema parity audit を実装する。

## 現在の状態

- Route: ml_model
- 状態: Kaggle train v2 / inference v3 完了、schema parity pass、提出なし
- CV: なし
- LB: なし
- 提出: なし

## コマンドログ

### 実行済み

```bash
uv run python scripts/new_steering.py --experiment exp145_learned_likelihood_rawtest_feature_generator_parity
uv run python scripts/new_experiment.py --name exp145_learned_likelihood_rawtest_feature_generator_parity --source templates/experiment
python -m py_compile experiments/exp145_learned_likelihood_rawtest_feature_generator_parity/learned_likelihood_rawtest_feature_generator_parity.py
uv run python scripts/validate_experiment.py --experiment exp145_learned_likelihood_rawtest_feature_generator_parity
uv run ruff check experiments/exp145_learned_likelihood_rawtest_feature_generator_parity/learned_likelihood_rawtest_feature_generator_parity.py
uv run ruff format --check experiments/exp145_learned_likelihood_rawtest_feature_generator_parity/learned_likelihood_rawtest_feature_generator_parity.py
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp145_learned_likelihood_rawtest_feature_generator_parity --notebook train --kernel-id kentookumura/exp145-train --title 'exp145 train' --run-on-push --strict
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp145_learned_likelihood_rawtest_feature_generator_parity --notebook inference --kernel-id kentookumura/exp145-inference --title 'exp145 inference' --run-on-push --strict
task push-kaggle-train EXP=exp145_learned_likelihood_rawtest_feature_generator_parity
kaggle kernels output kentookumura/exp145-train -p experiments/exp145_learned_likelihood_rawtest_feature_generator_parity/kaggle/output/train_v2
task push-kaggle-infer EXP=exp145_learned_likelihood_rawtest_feature_generator_parity
kaggle kernels output kentookumura/exp145-inference -p experiments/exp145_learned_likelihood_rawtest_feature_generator_parity/kaggle/output/inference_v3
```

### 失敗

```bash
uv run python experiments/exp145_learned_likelihood_rawtest_feature_generator_parity/learned_likelihood_rawtest_feature_generator_parity.py --mode train --max-rows 200 --output-dir /tmp/exp145_smoke
```

ローカル環境に `lightgbm` がなく、exp111 保存済み booster のロード前に `ModuleNotFoundError: No module named 'lightgbm'` で停止した。Kaggle runtime では LightGBM 前提のため、end-to-end smoke は Kaggle 実行で確認する。

### Kaggle 実行結果

- train v1: exp111 LightGBM model の期待 48 feature に対し generator が 32 feature しか渡さず失敗。exp111 の candidate-long 16列を復元して修正した。
- train v2: 完了。full-train `ml_features` は 3,783,989 rows / 773 wells、schema columns 51、schema parity pass、mismatch rows 0。
- inference v1: raw data path が Kaggle input mount に解決されず失敗。`ExperimentPaths().raw_data_dir` を使うよう修正した。
- inference v2: raw-test replay に exp099 `multiobs_*` 列がなく失敗。exp099 helper を同梱し、raw-test prefix GR から target-free multiobs を生成するよう修正した。
- inference v3: 完了。raw-test `ml_features` は 14,151 rows / 3 wells、schema columns 51、schema parity pass、mismatch rows 0。

### 生成物

- Train output: `experiments/exp145_learned_likelihood_rawtest_feature_generator_parity/kaggle/output/train_v2`
- Inference output: `experiments/exp145_learned_likelihood_rawtest_feature_generator_parity/kaggle/output/inference_v3`
- full-train `ml_features` decompressed SHA: `e1c276d69e9355f6c03c18ac51a0883ee99ec6d80d040a5c62e5d55048bb7456`
- raw-test `ml_features` decompressed SHA: `61a21bb1b52eb8ae2d242c758732fe3cb10682d9d8b147ebe4a40f75419704c8`
- raw-test likelihood long decompressed SHA: `4b50d801be8d3e0977b6699eea5110321d55df15b9ccfa46998a02d1f8b3fdf6`
- feature schema SHA: `b1285777136304d65c927d28a1d0f57d68c0e45a9c4d8a0cbaaff054e4315cf8`
- schema parity SHA: `737455382dad20f6e94a6c196be2a2ed45028ec22eacebadc9641dca5249f2b0`

## 変更点

- `learned_likelihood_rawtest_feature_generator_parity.py` を追加。
- `public_notebook_replay_audit.py` を同梱。
- train notebook を full-train generator 入口に更新。
- inference notebook を raw-test generator 入口に更新。
- config / README / result / metrics / steering を更新。

## 再現性メモ

- seed policy: exp145 自体は新規 RNG なし。
- stochastic components: 上流 exp111 LightGBM 学習、上流 exp072 PF/Beam replay。
- CPU/GPU runtime: CPU。
- Kaggle kernel id / version: train `kentookumura/exp145-train` v2、inference `kentookumura/exp145-inference` v3。
- input / feature schema SHA: `b1285777136304d65c927d28a1d0f57d68c0e45a9c4d8a0cbaaff054e4315cf8`。
- feature content SHA: full-train / raw-test の decompressed SHA を上記に記録。
- model manifest / model SHA: classifier `c4c65558ae07fc74735d7c41f7cdc605350112409273aa314cfb0122ed1e9f29`、expected-error `308242bf901c3db167e97b4750d389aa5b69cab492fe61cff2eeff82133725f3`、manifest `178e8b3124b817a2b230080fc041aaaee1b06941e5a4223a68cc31bf26e68010`。
- prediction SHA: 対象外。
- submission SHA: 対象外。
- rerun check: 未実行。

## 次のアクション

1. exp145 cache を使い、exp127 の learned likelihood feature family を exp092 full-row add-only 実験として再評価する。
2. 改善しても direct submit せず、worst-well regression、exp115 hidden-like stress、raw-test inference flow を確認する。
