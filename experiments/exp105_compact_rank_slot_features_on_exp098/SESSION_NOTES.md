# exp105_compact_rank_slot_features_on_exp098 セッションノート

## 目的

exp098 の rank-slot structured features から、重複・符号反転・低 utility の列を削った compact 版を評価する。候補値を直接選択または平均せず、exp073/exp072 base surface に add-only の補助特徴量として渡す。

## 現在の状態

- Route: ml_model
- 状態: completed_train_side_rejected
- CV: best `lgb2` pooled RMSE 9.441103161
- LB: 未提出
- inference: 未選択。train-side OOF の確認まで停止。

## コマンドログ

### 実行済み

```bash
make new-steering EXP=exp105_compact_rank_slot_features_on_exp098
make new-exp EXP=exp105_compact_rank_slot_features_on_exp098 SOURCE=experiments/exp098_selector_rank_slot_features_on_exp073
.venv/bin/python -m py_compile experiments/exp105_compact_rank_slot_features_on_exp098/compact_rank_slot_features_on_exp098.py
.venv/bin/python -m json.tool experiments/exp105_compact_rank_slot_features_on_exp098/exp105_compact_rank_slot_features_on_exp098_train.ipynb
.venv/bin/python -m json.tool experiments/exp105_compact_rank_slot_features_on_exp098/exp105_compact_rank_slot_features_on_exp098_inference.ipynb
make validate-exp EXP=exp105_compact_rank_slot_features_on_exp098
make prepare-kaggle-notebooks EXP=exp105_compact_rank_slot_features_on_exp098 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp105-compact-rank-slot-features-on-exp098-train --title 'exp105 compact rank slot features on exp098 train' --run-on-push --strict"
make push-kaggle-train EXP=exp105_compact_rank_slot_features_on_exp098
kaggle kernels pull kentookumura/exp105-compact-rank-slot-features-on-exp098-train -p /tmp/kaggle-pull/exp105-compact-rank-slot-features-on-exp098-train -m
kaggle kernels logs kentookumura/exp105-compact-rank-slot-features-on-exp098-train
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp105-compact-rank-slot-features-on-exp098-train
kaggle kernels output kentookumura/exp105-compact-rank-slot-features-on-exp098-train -p experiments/exp105_compact_rank_slot_features_on_exp098/kaggle/output/train_v1
gzip -dc experiments/exp105_compact_rank_slot_features_on_exp098/kaggle/output/train_v1/artifacts/exp105_compact_rank_slot_features_on_exp098_predictions.csv.gz | sha256sum
```

実装内容:

- `docs/legacy/steering/20260622-exp105-compact-rank-slot-features-on-exp098/` を作成した。
- exp098 から実験フォルダをコピーし、notebook / settings / config / 実装名を exp105 に更新した。
- `compact_rank_slot_features_on_exp098.py` に `rank_slot_compact` feature group を追加した。
- active variant を `compact_rank_slot_features` のみにした。
- inference notebook は train-side OOF レビュー前に明示停止する。
- py_compile、notebook JSON check、`make validate-exp` は通過した。
- Kaggle train package は `experiments/exp105_compact_rank_slot_features_on_exp098/kaggle/train` に生成済み。
- `kernel-metadata.json` は GPU 有効、internet 無効、kernel source `kentookumura/exp072-exp063-full-replay-feature-cache-train`、run_on_push true。
- synthetic frame check で `rank_slot_compact` は 22 列であることを確認した。
- Kaggle train kernel v1 push 成功。URL: https://www.kaggle.com/code/kentookumura/exp105-compact-rank-slot-features-on-exp098-train
- `kaggle kernels pull ... -m` は成功し、同じ kernel id の存在を確認した。
- 初期 `kaggle kernels logs` は空。3分 follow を開始したが、ユーザー指示によりローカル監視プロセスを停止した。Kaggle 側の実行は継続中として扱い、完了連絡待ち。
- ユーザーから完了連絡後に logs と output を取得した。
- Kaggle train v1 は `train_completed`。runtime は 11,443.651 sec。
- features: 218 = base 196 + compact rank-slot 22。
- pooled OOF: `lgb2` 9.441103161、`lgb1` 9.477699412、`lgb_mean` 9.506397523、`lgb0` 9.774440354。
- best `lgb2` は exp098 `lgb1` 9.358151052 から +0.082952 悪く、exp098 `lgb_mean` 9.427447987 から +0.013655 悪い。exp092 `lgb1` 9.322479896 との差は +0.118623。
- Rank1 source distribution は `pf_ancc` 33.65%、`beam_mean` 24.55%、`likpf_mean` 41.80%。`sc_ens` / `hyb` は rank1 / rank2 では 0。
- compact feature set は rejected。提出しない。

### 未実行

なし。

## 変更点

- base 196 features は exp098 と同じ exp073/exp072 surface を使う。
- compact rank-slot group は 22 列に限定する。
- 残す列: per-slot delta / score / source_code / U-space slope / curvature / resid_mad、score entropy、top1 margin、top-3 U std/range。
- 削る列: pairwise candidate delta、rank 間 `u_diff` / `u_absdiff`、`u_corr` と `u_resid` の符号反転ペア、`u_fit_degree`、`rank*_is_*` flags。

## 再現性メモ

- seed policy: fixed GroupKFold seed; rank-slot feature generation has no RNG.
- stochastic components: upstream exp072 PF/Beam cache、GPU LightGBM training。
- CPU/GPU runtime: primary `gpu_repro_guard_dp_threads8`。
- deterministic anchor ではなく train-side feature audit として扱う。
- input cache SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`。
- model manifest SHA: `ae6370fbbe8d1bc37ffdfb7f3c9ad6683bc2b38e3e7f1af72b7c57c840c1a2ca`。
- predictions gzip SHA: `397e24aeb15d550ad6fb5c58f8eb4bd462d3c4047d16042ed6cc3b882f9daa0c`。
- predictions decompressed SHA: `610154188ad092b8fed9dc60699aa797bd7397e92448d7b5068b6b273fcb374d`。
- `lgb2` prediction SHA: `fbe28f97011ce933aa619e500292d4a403d9f457e5100ab77765b8f9028dbe2f`。

## 次のアクション

1. exp105 compact 22-column set は rejected として閉じる。
2. exp098 の全 rank-slot features を rank-slot 比較基準として維持する。
3. rank-slot を次に使う場合は、exp092 への add-only merge または top-n candidate-only の別 ablation で検証する。
