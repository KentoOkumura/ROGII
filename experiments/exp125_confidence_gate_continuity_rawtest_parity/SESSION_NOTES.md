# exp125_confidence_gate_continuity_rawtest_parity セッションノート

## 目的

`exp102` best gate、`exp112` expected-error gate、optional dense/high-drift gate を continuity / worst-well / raw-test parity の観点で比較し、直接 inference port に進めるか、ML feature / segment selector 診断へ戻すかを判断する。

## 現在の状態

- Route: pf_beam
- 状態: completed_train_side_audit_no_submit
- CV: fair shared best RMSE 11.540333945
- LB: まだなし
- 提出候補: なし

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 実行済み

```bash
make new-steering EXP=exp125_confidence_gate_continuity_rawtest_parity
make new-exp EXP=exp125_confidence_gate_continuity_rawtest_parity
.venv/bin/python -m py_compile experiments/exp125_confidence_gate_continuity_rawtest_parity/confidence_gate_continuity_rawtest_parity.py
make validate-exp EXP=exp125_confidence_gate_continuity_rawtest_parity
make prepare-kaggle-notebooks EXP=exp125_confidence_gate_continuity_rawtest_parity EXTRA_ARGS="--notebook train --run-on-push --strict"
make prepare-kaggle-notebooks EXP=exp125_confidence_gate_continuity_rawtest_parity EXTRA_ARGS="--notebook inference --strict"
make push-kaggle-train EXP=exp125_confidence_gate_continuity_rawtest_parity
make prepare-kaggle-notebooks EXP=exp125_confidence_gate_continuity_rawtest_parity EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp125-cg-continuity-train --title exp125-cg-continuity-train --run-on-push --strict"
make push-kaggle-train EXP=exp125_confidence_gate_continuity_rawtest_parity
kaggle kernels pull kentookumura/exp125-cg-continuity-train -p /tmp/kaggle-pull/exp125-cg-continuity-train -m
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp125-cg-continuity-train
kaggle kernels logs kentookumura/exp125-cg-continuity-train
kaggle kernels output kentookumura/exp125-cg-continuity-train -p experiments/exp125_confidence_gate_continuity_rawtest_parity/kaggle/output/train_v1
kaggle kernels status kentookumura/exp125-cg-continuity-train
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp125-cg-continuity-train
kaggle kernels output kentookumura/exp125-cg-continuity-train -p experiments/exp125_confidence_gate_continuity_rawtest_parity/kaggle/output/train_v1
kaggle kernels status kentookumura/exp125-cg-continuity-train
```

### メモ

- 最初の `make push-kaggle-train` は default long id/title の slug mismatch で失敗した。
- 短い canonical id/title `kentookumura/exp125-cg-continuity-train` / `exp125-cg-continuity-train` で再生成し、Kaggle v1 push は成功した。
- push 時に `kentookumura/exp101-pf-candidate-ranker-train` は invalid kernel source として追加されなかった。そのため exp101 manifest / schema parity check は missing_required になった。
- `logs -f` は当初空だったが、kernel status は RUNNING。後続 polling で完了ログを取得した。
- final status は `KernelWorkerStatus.COMPLETE`。

### 予定

```bash
なし
```

## 変更点

- `confidence_gate_continuity_rawtest_parity.py` を追加した。
- train notebook を saved OOF posthoc audit 用に更新した。
- inference notebook は no-submission summary のみにした。
- config に exp102 / exp112 入力、shared surface、guardrail、raw-test parity checklist を記録した。

## 再現性メモ

- seed policy: `no_new_rng_posthoc_saved_oof_audit`
- stochastic components: 上流 exp099 PF/Beam cache、exp101 ranker、exp111/112 learned likelihood に依存。exp125 自体は RNG なし。
- CPU/GPU runtime: CPU-only、GPU 不要。
- Kaggle kernel id / version: `kentookumura/exp125-cg-continuity-train` v1 COMPLETE。
- input / feature schema SHA: exp102 OOF decompressed `469e9fa137...`、exp112 OOF decompressed `e3df222a...`、exp099 cache decompressed `1939d536...`、exp112 feature schema `b128577...`。
- feature content SHA: saved OOF gzip は decompressed content SHA を記録済み。
- model manifest / model SHA: exp101 manifest / schema は missing_required。新規 model はなし。
- prediction SHA: fair shared prediction content `1af3ae4980362fa246a5a61ac139ab74790bab957c7f2b8a4034a903141147e0`。
- submission SHA: submission を作らないため対象外。
- rerun check: 未実行。

## 次のアクション

1. `KAGGLE_DIRECTION.md` の backlog を整理する。
2. 必要なら exp112 feature cache の ML add-only 評価、または segment selector 側へ進める。
