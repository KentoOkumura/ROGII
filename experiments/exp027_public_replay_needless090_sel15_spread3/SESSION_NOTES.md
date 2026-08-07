# exp027_public_replay_needless090_sel15_spread3 セッションノート

## 目的

`public_notebook_catchup_after_self_improvements` で最初の replay 候補になった `needless090/lb8-781-rogii-sel15-spread3` を Kaggle 上で無改造 replay し、output / runtime / dependency / LB を記録する。

## 現在の状態

- 状態: Kaggle replay completed; CLI submit blocked by Kaggle API 400
- CV: まだなし
- LB: まだなし
- source notebook: `docs/notebooks/rogii-wellbore-geology-prediction/score_ascending_latest/needless090__lb8-781-rogii-sel15-spread3/lb8-781-rogii-sel15-spread3.ipynb`
- source metadata: CPU、internet off、external dataset/kernel/model sources なし
- title score: Public LB 8.781
- Kaggle kernel: `kentookumura/exp027-public-replay-needless090-sel15-spread3` version 1
- output path: `/tmp/kaggle-output/exp027_public_replay_needless090_sel15_spread3/inference/submission.csv`
- submission sha256: `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`
- submit-check: PASS
- competition submit: ref `53420592`
- Public LB: 8.781

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

- 2026-06-06: `uv run python scripts/new_steering.py --experiment exp027_public_replay_needless090_sel15_spread3` で steering docs を作成。
- 2026-06-06: `uv run python scripts/new_experiment.py --name exp027_public_replay_needless090_sel15_spread3` で実験フォルダを作成。
- 2026-06-06: public archive の `lb8-781-rogii-sel15-spread3.ipynb` を inference notebook として配置。
- 2026-06-06: `config.yaml`、README、SESSION_NOTES、result、metrics を replay 用に更新。
- 2026-06-06: `uv run python scripts/validate_experiment.py --experiment exp027_public_replay_needless090_sel15_spread3` が通過。
- 2026-06-06: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp027_public_replay_needless090_sel15_spread3 --notebook inference --kernel-id kentookumura/exp027-public-replay-needless090-sel15-spread3 --run-on-push --title 'exp027 public replay needless090 sel15 spread3' --strict` が通過。
- 2026-06-06: `kaggle kernels push -p experiments/exp027_public_replay_needless090_sel15_spread3/kaggle/inference` で version 1 を push。
- 2026-06-06: `kaggle kernels status kentookumura/exp027-public-replay-needless090-sel15-spread3` は Kaggle API 500。
- 2026-06-06: `kaggle kernels output kentookumura/exp027-public-replay-needless090-sel15-spread3 -p /tmp/kaggle-output/exp027_public_replay_needless090_sel15_spread3/inference` で output を取得。`submission.csv`、log、support files が取得できた。
- 2026-06-06: `uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp027_public_replay_needless090_sel15_spread3/inference/submission.csv` が PASS。
- 2026-06-06: 誤って `kaggle competitions submit -c rogii-wellbore-geology-prediction -f /tmp/.../submission.csv -m ...` の file upload 形式で提出し、upload 後 `400 Client Error: Bad Request`。Notebook-only code competition では kernel output 形式の `-k <kernel> -v <version> -f submission.csv` が必要。
- 2026-06-06: 同じ誤った file upload 形式を短い message `-m exp027` でも再試行したが同じ 400。提出履歴に ref は増えず。
- 2026-06-06: ユーザー確認で Kaggle kernel version 1 は successful と判明し、UI から手動 submit 済み。`kaggle competitions submissions rogii-wellbore-geology-prediction` で ref `53420592`、status `PENDING` を確認。
- 2026-06-06: `Taskfile.yml` と `Makefile` に `submit-code` task を追加。今後の正しい CLI は `task submit-code EXP=exp027_public_replay_needless090_sel15_spread3 KERNEL=kentookumura/exp027-public-replay-needless090-sel15-spread3 KERNEL_VERSION=1 OUTPUT_FILE=submission.csv MESSAGE="exp027"`。
- 2026-06-07: ユーザー確認後、`kaggle competitions submissions rogii-wellbore-geology-prediction` で ref `53420592`、status `COMPLETE`、Public LB 8.781 を確認。
- 2026-06-07: output `submission.csv` と Kaggle log を `artifacts/` に保存し、`record_submission.py` で `SUBMISSIONS.md` v012、`record_experiment.py` で metrics / summary を更新。
- 2026-06-07: ユーザーが再現性確認のため exp027 を再提出。`kaggle competitions submissions rogii-wellbore-geology-prediction` で ref `53449441`、status `PENDING` を確認し、`SUBMISSIONS.md` v015 に pending として記録。

### 予定

```bash
task validate-exp EXP=exp027_public_replay_needless090_sel15_spread3
task prepare-kaggle-notebooks EXP=exp027_public_replay_needless090_sel15_spread3 EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp027-public-replay-needless090-sel15-spread3 --run-on-push --title 'exp027 public replay needless090 sel15 spread3' --strict"
kaggle kernels push -p experiments/exp027_public_replay_needless090_sel15_spread3/kaggle/inference
kaggle kernels status kentookumura/exp027-public-replay-needless090-sel15-spread3
kaggle kernels output kentookumura/exp027-public-replay-needless090-sel15-spread3 -p /tmp/kaggle-output/exp027_public_replay_needless090_sel15_spread3/inference
task submit-check EXP=exp027_public_replay_needless090_sel15_spread3 SUBMISSION=/tmp/kaggle-output/exp027_public_replay_needless090_sel15_spread3/inference/submission.csv
task submit-code EXP=exp027_public_replay_needless090_sel15_spread3 KERNEL=kentookumura/exp027-public-replay-needless090-sel15-spread3 KERNEL_VERSION=1 OUTPUT_FILE=submission.csv MESSAGE="exp027"
```

## 変更点

- Public notebook replay. No self-model code changes.
- Inference notebook body is the archived public notebook.
- Train notebook is unused and kept only to satisfy repository experiment structure.

## 結果

- Kaggle runtime log: 約 204 秒で `submission.csv` 生成完了。
- output rows: 14,151
- prediction range: 11587.038593 - 12240.016066
- prediction mean: 11903.630073
- duplicate IDs: 0
- missing values: 0
- exp026 submission との差分: min -26.226663、max 7.229360、mean -3.672535、abs mean 5.777701、RMSE 8.098430、corr 0.999685
- CLI submit: file upload 形式は Kaggle API 400。UI submit は ref `53420592` / Public LB 8.781。

## 次のアクション

1. Public LB anchor は exp027 replay の 8.781 に更新済み。
2. 今後の CLI submit は `task submit-code ...` を使う。
3. 次は公開 replay route の追加確認か、自前 pseudo-tail 系への戻りを戦略判断する。
