# exp109_typewell_neighbor_prior_features セッションノート

## 目的

exp065 の native typewell overlap group を使い、同 group の train-fold wells から作る TVT drift prior が、exp099 の `likpf_mean` / `pf_ancc` / `beam_mean` 候補の誤差を補正できるかを train pseudo-tail OOF で確認する。

## 現在の状態

- Route: ensemble
- 状態: 実装中
- CV: まだなし
- LB: まだなし

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 予定

```bash
task validate-exp EXP=exp109_typewell_neighbor_prior_features
task prepare-kaggle-notebooks EXP=exp109_typewell_neighbor_prior_features EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp109-typewell-neighbor-prior-train --title 'exp109 typewell neighbor prior train' --run-on-push --strict"
task push-kaggle-train EXP=exp109_typewell_neighbor_prior_features
```

## 変更点

- `docs/legacy/steering/20260622-exp109-typewell-neighbor-prior-features/` を作成。
- `experiments/exp109_typewell_neighbor_prior_features/` を作成。
- `config.yaml` を exp065 / exp099 入力の train-side audit に更新。
- `typewell_neighbor_prior_features.py` を追加。
  - exp099 v2 feature cache を読み、`true_tvt = last_known_tvt + target` を scoring 用に復元。
  - exp065 cluster assignments から `native_overlap=1`、`native_overlap=0.999`、`exact_hash` group を読む。
  - well-grouped folds で、valid well の prior source を train-fold same-cluster wells のみに制限。
  - `md_since` 軸で neighbor `true_tvt - last_known_tvt` curve を補間し、median drift を prior にする。
  - `likpf_mean` / `pf_ancc` / `beam_mean` へ `alpha * clipped(prior - base)` の後段補正を作る。
- train notebook を設定確認、入力確認、audit 実行、metrics 表示の構成に更新。
- inference notebook は train-side audit only guard に更新。
- Kaggle train v1 push は成功したが、`kentookumura/exp065-typewell-supertype-cluster-cv-audit` が無効な kernel source として拒否された。正しい source は `kentookumura/exp065-typewell-supertype-cluster-cv-audit-train` だったため、config を修正して v2 を再 push する。
- Kaggle train v2 push 完了。kernel: `kentookumura/exp109-typewell-neighbor-prior-train`。`kaggle kernels status` では `KernelWorkerStatus.RUNNING`。`logs -f` / 通常 `logs` はこの時点では空。ユーザー指示により監視は停止。
- Kaggle train v2 完了。output を `experiments/exp109_typewell_neighbor_prior_features/kaggle/output/train_v2` に取得。
- best は `native_overlap_0p999_likpf_mean_corr_a0p2_c40`。RMSE 11.143359521 / MAE 7.025321534 / within10 0.779883345。`likpf_mean` RMSE 11.594897672 / within10 0.772807479 から RMSE -0.451538151、within10 +0.007075866。
- 全 distance bucket で `likpf_mean` より RMSE 改善。well 単位は 413 改善 / 345 悪化 / 15 同値。最大悪化 +6.594183 RMSE、最大改善 -6.447593 RMSE。

## 再現性メモ

- seed policy: deterministic well fold assignment with fixed seed 42
- stochastic components: 新規なし。上流 exp072 / exp099 / exp065 artifacts は固定入力。
- CPU/GPU runtime: CPU, GPU disabled
- Kaggle kernel id / version: `kentookumura/exp109-typewell-neighbor-prior-train` v2
- input / feature schema SHA: exp099 raw `4bd9df60f5c09f7a3029dac399afef73aa45b0158a7fd06a62a56f85fd0fde38` / decompressed `1939d536b1e56f7c0ea3847cc386ef769b0d33759d16e816c9ce180f0532df9a` / schema `203e4f9a280fe901f5f21d39b85c3e0e2a7fe10c466081c15015c7fb014a0413`
- cluster assignment SHA: `dcda8588cc1dd9261bafae7de00c890393e38b8a0ca0eb86fbba18a2cffc4a50`
- feature content SHA: OOF raw `cc4c017baff6410ce8a0cf52e0a0e76d6f7309fd9b163eea43c0173b7e5fb660` / decompressed `ec1e105a3021badf7441768329e94dfce874f110927bf9af1a3968ddbc609e29`
- model manifest / model SHA: model なし
- prediction SHA: OOF prediction SHA のみ記録。submission なし
- submission SHA: submission なし
- rerun check: 未実行

## 次のアクション

1. `typewell_neighbor_prior_rawtest_parity_gate` として raw-test parity / worst-well gate を設計する。
2. exp109 は train-side audit 完了として記録し、直接 inference port / submit はしない。
