# exp128_trajectory_local_typewell_self_gr_switch_audit セッションノート

## 目的

`trajectory_local_typewell_self_gr_switch_audit` バックログの実装。exp099 の PF/Beam / likelihood-PF 候補を基準にし、同一 trajectory 内の局所窓で typewell GR cost と same-horizontal visible-prefix self-GR cost を比較する。self-GR が target-free に十分優勢な窓だけで hard switch / soft blend 候補を作り、提出候補ではなく train-side 診断として評価する。

## 現在の状態

- Route: ensemble
- 状態: completed_train_side_rejected_no_submit
- CV: 11.594897672217703
- LB: まだなし

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 2026-06-25 実装

```bash
make new-steering EXP=exp128_trajectory_local_typewell_self_gr_switch_audit
make new-exp EXP=exp128_trajectory_local_typewell_self_gr_switch_audit
python3 -m py_compile experiments/exp128_trajectory_local_typewell_self_gr_switch_audit/trajectory_local_typewell_self_gr_switch_audit.py
python3 - <<'PY'
import yaml
from pathlib import Path
p = Path('experiments/exp128_trajectory_local_typewell_self_gr_switch_audit/config.yaml')
yaml.safe_load(p.read_text())
print('yaml ok')
PY
```

### 2026-06-25 Kaggle train push

最初に次の id/title で prepare/push した。

```bash
make prepare-kaggle-notebooks EXP=exp128_trajectory_local_typewell_self_gr_switch_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp128-trajectory-local-switch-train --title 'exp128 train' --run-on-push --strict"
make push-kaggle-train EXP=exp128_trajectory_local_typewell_self_gr_switch_audit
```

Kaggle CLI は push 成功と同時に title slug mismatch warning を出し、実際の URL は `https://www.kaggle.com/code/kentookumura/exp128-train` になった。ログ監視はユーザー指示で停止した。

希望名に合わせるため、repo-local skill の Kaggle 手順を修正し、`kernel-id` 末尾 slug と `title` 由来 slug を一致させるルールを明記した。その後、同じ exp のまま次で再 prepare/push した。

```bash
make prepare-kaggle-notebooks EXP=exp128_trajectory_local_typewell_self_gr_switch_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp128-trajectory-local-switch-train --title 'exp128 trajectory local switch train' --run-on-push --strict"
make push-kaggle-train EXP=exp128_trajectory_local_typewell_self_gr_switch_audit
```

新 kernel は warning なしで version 1 が push された。

- Kernel: `kentookumura/exp128-trajectory-local-switch-train`
- URL: https://www.kaggle.com/code/kentookumura/exp128-trajectory-local-switch-train
- 実行監視: ユーザー指示により未実施。

### 2026-06-25 Kaggle train v1 完了確認

ユーザーから完了連絡を受け、logs と output を取得した。

```bash
kaggle kernels logs kentookumura/exp128-trajectory-local-switch-train
kaggle kernels output kentookumura/exp128-trajectory-local-switch-train -p experiments/exp128_trajectory_local_typewell_self_gr_switch_audit/kaggle/output/train_v1
```

v1 は 3,783,989 rows / 773 wells で完了し、生成物は `kaggle/output/train_v1/artifacts/` に保存された。しかし soft blend 候補の best は coverage 0.756345750 で、baseline `likpf_mean` の coverage 1.0 と比較面が一致していなかった。原因は `gate=0` の row でも `(1-gate)*base + gate*self_prior` が `0 * NaN` を通じて NaN になり、self prior が finite な subset だけで RMSE 評価されていたこと。

- v1 見かけ best: `likpf_mean_local_self_gr_blend_gap0p15_w0p25` RMSE 11.552085575 / coverage 0.756345750。
- baseline: `likpf_mean` RMSE 11.594897672 / coverage 1.0。
- hard switch 系は switch rate 0 で baseline と同値。

結論: v1 metrics は invalid。soft blend 実装を base copy + positive gate row のみ上書きに修正し、v2 を同じ kernel id に再 push する。

### 2026-06-25 Kaggle train v2 push

soft blend の NaN 伝播を修正し、検証後に同じ kernel id へ v2 として push した。

```bash
make validate-exp EXP=exp128_trajectory_local_typewell_self_gr_switch_audit
make prepare-kaggle-notebooks EXP=exp128_trajectory_local_typewell_self_gr_switch_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp128-trajectory-local-switch-train --title 'exp128 trajectory local switch train' --run-on-push --strict"
make push-kaggle-train EXP=exp128_trajectory_local_typewell_self_gr_switch_audit
```

- Kernel: `kentookumura/exp128-trajectory-local-switch-train`
- Version: 2
- URL: https://www.kaggle.com/code/kentookumura/exp128-trajectory-local-switch-train
- Status: pushed / running

### 2026-06-26 Kaggle train v2 完了確認

ユーザーから完了連絡を受け、logs と output を取得した。

```bash
kaggle kernels logs kentookumura/exp128-trajectory-local-switch-train
kaggle kernels output kentookumura/exp128-trajectory-local-switch-train -p experiments/exp128_trajectory_local_typewell_self_gr_switch_audit/kaggle/output/train_v2
```

v2 は 3,783,989 rows / 773 wells で完了。v1 の soft blend coverage bug は修正され、全 switch/blend 候補が coverage 1.0 で評価された。正式結果は改善なし。

- best candidate: `likpf_mean`
- RMSE: 11.594897672217703
- MAE: 7.067632584311985
- within10: 0.772807479091509
- delta vs `likpf_mean`: 0.0
- finite self prior rate: 0.7563457504765474
- mean `typewell_cost - self_cost`: -0.7429656386375427
- switch/blend gate: 0.0

解釈: self-GR prefix match は typewell observation cost より弱く、保守的な gap 条件では一度も発火しない。`self_gr_prefix_prior_tvt` 単体は worst-well で数千 ft 規模に壊れるため、trajectory local switch / blend としては棄却する。推論化・提出なし。

### 予定

```bash
task validate-exp EXP=exp128_trajectory_local_typewell_self_gr_switch_audit
task kaggle-logs KERNEL=kentookumura/exp128-trajectory-local-switch-train
```

## 変更点

- `trajectory_local_typewell_self_gr_switch_audit.py` を追加。
- train notebook を、設定確認、入力確認、監査実行、生成物確認のセル構成に更新。
- inference notebook は診断専用として明示的に停止する形に更新。
- `config.yaml` に route、親実験、local switch 閾値、出力生成物、再現性方針を記録。

## 再現性メモ

- seed policy: deterministic single-process window scan。新規処理自体は RNG 不使用。
- stochastic components: 上流 exp099 PF/Beam / likelihood-PF cache。
- CPU/GPU runtime: CPU-only、GPU 不使用。
- Kaggle kernel id / version: `kentookumura/exp128-trajectory-local-switch-train` v2 completed after v1 invalid soft-blend coverage bug。
- input / feature schema SHA: Kaggle train 実行時に summary JSON へ記録。
- feature content SHA: exp099 gzip cache は decompressed SHA を主証拠として記録。
- model manifest / model SHA: 学習モデルなし。
- prediction SHA: OOF gzip raw SHA `972ac359b8266399c2978822cded25677dd2516ffb584bd79913a1c334c29bc6`、decompressed SHA `fa274d49641bfbf033817f79971e20a9678499336b449d312601a6066bfa5731`。
- submission SHA: 提出なし。
- rerun check: v1 は invalid、v2 を正式結果として記録。採用候補なし。

## 次のアクション

1. `experiment_summary.md` と `KAGGLE_DIRECTION.md` に反映する。
2. self-GR local switch は閉じ、self-GR 由来情報は補助 confidence に限定する。
