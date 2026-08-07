# exp028_public_replay_second_sel15_or_blend_audit セッションノート

## 目的

`public_notebook_catchup_after_self_improvements` の2本目 replay 候補として `needless090/lb-8-860-rogii-sel15-256seeds` を Kaggle 上で無改造 replay できる状態にし、exp027 との差分、runtime、dependency、LB を記録する。blend は exp028 の output と LB が揃ってから別判断にする。

## 現在の状態

- 状態: Kaggle replay completed; not submitted because output is identical to exp027
- CV: まだなし
- LB: まだなし
- source notebook: `docs/notebooks/rogii-wellbore-geology-prediction/score_ascending_latest/needless090__lb-8-860-rogii-sel15-256seeds/lb-8-860-rogii-sel15-256seeds.ipynb`
- source metadata: CPU、internet off、external dataset/kernel/model sources なし
- title score: Public LB 8.860
- Kaggle kernel: `kentookumura/exp028-second-sel15-replay` version 2
- output path: `/tmp/kaggle-output/exp028_public_replay_second_sel15_or_blend_audit/inference/submission.csv`
- submission sha256: `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`
- submit-check: PASS
- competition submit: 未実行
- Public LB: 未提出。exp027 と同一 submission のため、提出すれば exp027 と同じ 8.781 になる見込み。

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

- 2026-06-07: `uv run python scripts/new_steering.py --experiment exp028_public_replay_second_sel15_or_blend_audit` で steering docs を作成。
- 2026-06-07: `uv run python scripts/new_experiment.py --name exp028_public_replay_second_sel15_or_blend_audit --source experiments/exp027_public_replay_needless090_sel15_spread3` で実験フォルダを作成。
- 2026-06-07: public archive の `lb-8-860-rogii-sel15-256seeds.ipynb` を inference notebook として配置。
- 2026-06-07: `config.yaml`、README、SESSION_NOTES、result、metrics を exp028 replay 用に更新。
- 2026-06-07: `uv run python scripts/validate_experiment.py --experiment exp028_public_replay_second_sel15_or_blend_audit` が通過。
- 2026-06-07: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp028_public_replay_second_sel15_or_blend_audit --notebook inference --kernel-id kentookumura/exp028-public-replay-second-sel15-or-blend-audit --run-on-push --title 'exp028 public replay second sel15 or blend audit' --strict` が通過。
- 2026-06-07: `uv run python scripts/update_experiment_summary.py` で `experiment_summary.md` に exp028 を追加。
- 2026-06-07: `kaggle kernels push -p experiments/exp028_public_replay_second_sel15_or_blend_audit/kaggle/inference` で長い kernel id を push。CLI は成功表示だったが `kernels list` に出ず、output も空だった。
- 2026-06-07: canonical slug を安定させるため、`uv run python scripts/prepare_kaggle_notebooks.py --experiment exp028_public_replay_second_sel15_or_blend_audit --notebook inference --kernel-id kentookumura/exp028-second-sel15-replay --run-on-push --title 'exp028 second sel15 replay' --strict` で再生成。
- 2026-06-07: `kaggle kernels push -p experiments/exp028_public_replay_second_sel15_or_blend_audit/kaggle/inference` で `kentookumura/exp028-second-sel15-replay` version 2 を push。
- 2026-06-07: `kaggle kernels status kentookumura/exp028-second-sel15-replay` は Kaggle API 500。
- 2026-06-07: `kaggle kernels logs kentookumura/exp028-second-sel15-replay` で runtime log を取得。約384秒で `Done: 14151 rows`、約394秒で HTML conversion 完了。
- 2026-06-07: `kaggle kernels output kentookumura/exp028-second-sel15-replay -p /tmp/kaggle-output/exp028_public_replay_second_sel15_or_blend_audit/inference` で output を取得。
- 2026-06-07: `uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp028_public_replay_second_sel15_or_blend_audit/inference/submission.csv` が PASS。
- 2026-06-07: exp027 submission と比較し、diff min/max/mean/abs mean/RMSE がすべて 0、corr 1.000000。exp028 は提出しない判断。

### 実行済み主要コマンド

```bash
uv run python scripts/validate_experiment.py --experiment exp028_public_replay_second_sel15_or_blend_audit
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp028_public_replay_second_sel15_or_blend_audit --notebook inference --kernel-id kentookumura/exp028-second-sel15-replay --run-on-push --title 'exp028 second sel15 replay' --strict
kaggle kernels push -p experiments/exp028_public_replay_second_sel15_or_blend_audit/kaggle/inference
kaggle kernels logs kentookumura/exp028-second-sel15-replay
kaggle kernels output kentookumura/exp028-second-sel15-replay -p /tmp/kaggle-output/exp028_public_replay_second_sel15_or_blend_audit/inference
uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp028_public_replay_second_sel15_or_blend_audit/inference/submission.csv
```

## 変更点

- Public notebook replay. No self-model code changes.
- Inference notebook body is the archived public notebook.
- Train notebook is unused and kept only to satisfy repository experiment structure.
- Blend is explicitly disabled in config until exp028 output and LB are known.

## 結果

- Kaggle runtime log: 約384秒で `submission.csv` 生成完了。
- output rows: 14,151
- prediction range: 11587.038593 - 12240.016066
- prediction mean: 11903.630073
- duplicate IDs: 0
- missing values: 0
- SHA256: `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`
- exp027 submission との差分: min 0.000000、max 0.000000、mean 0.000000、abs mean 0.000000、RMSE 0.000000、corr 1.000000
- submit: 未実行。exp027 と同一なので提出 quota を使わない。

## 次のアクション

1. exp028 は追加 submit しない。
2. public replay route は exp027 を固定 anchor として扱う。
3. 次は自前 route に戻るか、別 family の public replay を artifact/dependency audit してから検討する。
