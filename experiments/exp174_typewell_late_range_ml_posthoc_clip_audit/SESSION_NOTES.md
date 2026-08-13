# exp174_typewell_late_range_ml_posthoc_clip_audit セッションノート

## 目的

`backlog/KAGGLE_DIRECTION.md` の `typewell_late_range_ml_posthoc_clip_audit` を実装する。`exp148` の保存済み OOF prediction に対して、typewell range 内位置に基づく条件付き lower-bound shrink / clip を no-training で監査する。

## 現在の状態

- Route: ml_model
- 状態: 完了、不採用、提出なし
- CV: baseline 8.501281182。発火 policy はすべて悪化
- LB: まだなし

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 予定

```bash
make prepare-kaggle-notebooks EXP=exp174_typewell_late_range_ml_posthoc_clip_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp174-typewell-late-range-ml-posthoc-clip-audit-train --title 'exp174 typewell late range ml posthoc clip audit train' --run-on-push --strict"
make push-kaggle-train EXP=exp174_typewell_late_range_ml_posthoc_clip_audit
kaggle kernels logs kentookumura/exp174-typewell-late-range-ml-posthoc-clip-audit-train
```

### 2026-07-03 実装

```bash
make new-steering EXP=exp174_typewell_late_range_ml_posthoc_clip_audit
make new-exp EXP=exp174_typewell_late_range_ml_posthoc_clip_audit
```

### 2026-07-03 Kaggle push 1

```bash
make push-kaggle-train EXP=exp174_typewell_late_range_ml_posthoc_clip_audit
```

- 結果: `SaveKernel` 400 で失敗。
- 原因: `kernel_sources` に存在しない長い slug を入れていた。
- 修正: `kentookumura/exp148-train`、`kentookumura/exp092-uproj-corr-disagree-train`、`kentookumura/exp073-full-replay-repro-guard-train` に変更。同じ canonical exp174 kernel id で再 prepare / push する。

### 2026-07-03 Kaggle push 2

```bash
make prepare-kaggle-notebooks EXP=exp174_typewell_late_range_ml_posthoc_clip_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp174-typewell-late-range-ml-posthoc-clip-audit-train --title 'exp174 typewell late range ml posthoc clip audit train' --run-on-push --strict"
make push-kaggle-train EXP=exp174_typewell_late_range_ml_posthoc_clip_audit
kaggle kernels pull kentookumura/exp148-train -p /tmp/kaggle-pull/exp148-train -m
kaggle kernels pull kentookumura/exp092-uproj-corr-disagree-train -p /tmp/kaggle-pull/exp092-uproj-corr-disagree-train -m
kaggle kernels pull kentookumura/exp073-full-replay-repro-guard-train -p /tmp/kaggle-pull/exp073-full-replay-repro-guard-train -m
```

- 結果: 再 push も `SaveKernel` 400 で失敗。
- 確認: 3つの source kernel は `kaggle kernels pull -m` で存在確認済み。
- 次の切り分け: 実行に必須なのは exp148 OOF だけなので、`kernel_sources` を `kentookumura/exp148-train` のみに最小化して同じ canonical exp174 kernel id で再 prepare / push する。

### 2026-07-03 Kaggle push 3

```bash
make prepare-kaggle-notebooks EXP=exp174_typewell_late_range_ml_posthoc_clip_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp174-typewell-late-range-ml-posthoc-clip-audit-train --title 'exp174 typewell late range ml posthoc clip audit train' --run-on-push --strict"
make push-kaggle-train EXP=exp174_typewell_late_range_ml_posthoc_clip_audit
```

- 結果: `kernel_sources` を exp148 のみにしても `SaveKernel` 400。
- payload: notebook 約 37KB、support module 約 35KB。サイズ起因ではない。
- id/title slug: `exp174-typewell-late-range-ml-posthoc-clip-audit-train` で一致、slug 長 54。
- 次の切り分け: Kaggle 側の slug/title 長制限の可能性が高いため、同じ exp174 のまま `exp174-typewell-late-clip-train` / `exp174 typewell late clip train` に短縮して再 prepare / push する。

### 2026-07-03 Kaggle push 4

```bash
make prepare-kaggle-notebooks EXP=exp174_typewell_late_range_ml_posthoc_clip_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp174-typewell-late-clip-train --title 'exp174 typewell late clip train' --run-on-push --strict"
make push-kaggle-train EXP=exp174_typewell_late_range_ml_posthoc_clip_audit
kaggle kernels logs kentookumura/exp174-typewell-late-clip-train
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp174-typewell-late-clip-train
```

- 結果: push 成功。Kaggle kernel v1: https://www.kaggle.com/code/kentookumura/exp174-typewell-late-clip-train
- 初回通常 logs と 3分の `logs -f` は空。Kaggle CLI の実行中ログ未反映パターンとして扱う。
- ユーザー指示により、こちらでの監視は停止。完了連絡後に同じ slug の logs / 必要なら output を確認する。

### 2026-07-03 Kaggle train v1 完了確認

```bash
kaggle kernels logs kentookumura/exp174-typewell-late-clip-train
kaggle kernels output kentookumura/exp174-typewell-late-clip-train -p /tmp/kaggle-output/exp174_typewell_late_range_ml_posthoc_clip_audit/train
```

- logs: 完了を確認。runtime は notebook log 上で約 2,160 sec。LightGBM config 0 / folds 0 / boosters 0 / control retraining none。
- output: OOF gzip 取得中に `kaggle kernels output` が code 137 で停止したが、小さい metrics CSV は取得できた。
- 実験配下に同期した小さい生成物:
  - `artifacts/exp174_typewell_late_range_ml_posthoc_clip_audit_candidate_metrics.csv`
  - `artifacts/exp174_typewell_late_range_ml_posthoc_clip_audit_bucket_metrics.csv`
  - `artifacts/exp174_typewell_late_range_ml_posthoc_clip_audit_by_well.csv`
  - `artifacts/exp174_typewell_late_range_ml_posthoc_clip_audit_changed_summary.csv`

主要結果:

- baseline `exp148_lgb_mean`: RMSE 8.501281182 / MAE 5.335651 / within10 0.856332
- lower bound `0.55/0.60/0.65` 系: changed_rows 0、baseline と同一
- best firing policy: `fixed_lb0p7_klp0p75_a0p25`、changed_rows 2,098 / 2 wells、RMSE 8.501891、baseline から +0.000609 悪化
- `fixed_lb0p7_klp0p8_a0p25`: changed_rows 1,325 / 1 well、RMSE 8.502021、+0.000740 悪化
- `known_last_m0p1_klp0p75_a0p25`: changed_rows 1,531 / 1 well、RMSE 8.502423、+0.001142 悪化
- 最大発火 `known_last_m0p05_klp0p75_a0p25`: changed_rows 13,657 / 14 wells、RMSE 8.518425、+0.017144 悪化

結論:

- exp148 ML 予測に対する typewell late-range lower-bound clip / shrink は不採用。
- inference port / submit はしない。
- typewell late-range prior を続ける場合も hard ML posthoc ではなく、PF/Beam candidate feature / selector prior に限定する。

## 変更点

- `docs/legacy/steering/20260703-exp174-typewell-late-range-ml-posthoc-clip-audit/` に requirements / design / tasklist を記入した。
- `config.yaml` を exp148 OOF primary の no-training posthoc audit 用に更新した。
- `typewell_late_range_ml_posthoc_clip_audit.py` を追加した。
- train notebook の正となる `exp174_typewell_late_range_ml_posthoc_clip_audit_train.py` を追加した。
- inference notebook は no-op guard とし、提出を作らない。

## Kaggle push 前の計算規模

- posthoc variant 数: `known_last_pct_min` 2 x (`fixed_lower_bounds` 4 + `known_last_margins` 4) x `alphas` 3 = 48
- LightGBM config 数: 0
- fold 数: 0
- booster 数: 0
- control 再学習: なし
- GPU: なし

## 再現性メモ

- seed policy: no_new_rng_posthoc_grid
- stochastic components: upstream exp148 LightGBM OOF predictions
- CPU/GPU runtime: CPU, GPU 不使用
- Kaggle kernel id / version: `kentookumura/exp174-typewell-late-clip-train` v1
- input / feature schema SHA: source summary JSON は output 取得が OOF gzip で code 137 停止したため未取得
- feature content SHA: 対象外。small metrics CSV の SHA を `result.md` に記録
- model manifest / model SHA: 対象外
- prediction SHA: selected OOF gzip は Kaggle output 取得が code 137 で停止したため未取得
- submission SHA: 対象外
- rerun check: 未実行

## 完了時の整理

- `backlog/KAGGLE_DIRECTION.md` から `typewell_late_range_ml_posthoc_clip_audit` を完了/不採用として外した。
- 関連する `typewell_late_range_pfbeam_candidate_prior`、`typewell_late_range_pfbeam_generation_soft_prior`、`typewell_late_range_clipped_candidate_augmentation` は exp174 の負例を反映して低優先 / hard clip 禁止の扱いに更新した。
- `experiment_summary.md` に exp174 の完了結果を記録した。
