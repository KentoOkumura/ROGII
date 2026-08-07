# exp130_pfbeam_normalized_diagnostic_score セッションノート

## 状態

- 2026-06-26: 実装済み。Kaggle train は未実行。
- Route: `ml_model`
- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- cache 親: `exp072_exp063_full_replay_feature_cache`

## 実装メモ

- `KAGGLE_DIRECTION.md` の `pfbeam_normalized_diagnostic_score` を実験化した。
- exp127 を派生元にしたが、exp112 learned likelihood feature cache 依存は削除した。
- exp072 full replay feature cache の `pf_ancc`、`pf_z`、`beam_mean_d`、`beam_med_d`、`likpf_mean_d` から candidate TVT を復元する。
- candidate TVT を `u = TVT + Z - (T0 + Z0)` に写し、well-local `u_scale` と `md_since_norm` で正規化する。
- known prefix の末尾 TVT_input/Z から `prefix_u_slope_per_md` を推定し、candidate path の normalized residual、roughness、curvature、candidate disagreement を作る。
- `pfbn_normalized_diagnostic_score` は row-wise instability の反転として作り、hard selector ではなく LightGBM add-only feature として使う。
- Inference notebook は no-submission summary のみを書き、`submission.csv` は作らない。
- 2026-06-27 の運用見直しにより、今後の再 push では `exp092_full_row_control` を無効化し、既存 exp092 metrics を baseline として参照する。control 再学習はユーザーの明示承認がある場合だけ行う。

## コマンド

```bash
make new-steering EXP=exp130_pfbeam_normalized_diagnostic_score
make new-exp EXP=exp130_pfbeam_normalized_diagnostic_score SOURCE=experiments/exp127_learned_likelihood_features_on_exp092
make prepare-kaggle-notebooks EXP=exp130_pfbeam_normalized_diagnostic_score EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp130-pfbn-score-train --title 'exp130 pfbn score train' --run-on-push --strict"
make push-kaggle-train EXP=exp130_pfbeam_normalized_diagnostic_score
```

## 検証

- `python3 -m py_compile experiments/exp130_pfbeam_normalized_diagnostic_score/pfbeam_normalized_diagnostic_score.py experiments/exp130_pfbeam_normalized_diagnostic_score/settings.py`: PASS
- `python3 -m json.tool experiments/exp130_pfbeam_normalized_diagnostic_score/exp130_pfbeam_normalized_diagnostic_score_train.ipynb`: PASS
- `python3 -m json.tool experiments/exp130_pfbeam_normalized_diagnostic_score/exp130_pfbeam_normalized_diagnostic_score_inference.ipynb`: PASS
- `.venv/bin/ruff check experiments/exp130_pfbeam_normalized_diagnostic_score/pfbeam_normalized_diagnostic_score.py experiments/exp130_pfbeam_normalized_diagnostic_score/settings.py`: PASS
- `make validate-exp EXP=exp130_pfbeam_normalized_diagnostic_score`: PASS
- 合成 frame 40 rows で `build_pfbeam_normalized_diagnostic_features()` smoke: PASS。87 features、finite check PASS。
- local 実データ smoke は exp072 cache が手元に無いため未実施。Kaggle input source を正とする。
- `make prepare-kaggle-notebooks EXP=exp130_pfbeam_normalized_diagnostic_score EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp130-pfbn-score-train --title 'exp130 pfbn score train' --run-on-push --strict"`: PASS
- `make prepare-kaggle-notebooks EXP=exp130_pfbeam_normalized_diagnostic_score EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp130-pfbn-score-infer --title 'exp130 pfbn score infer' --run-on-push --strict"`: PASS
- `make update-summary`: PASS。`experiment_summary.md` に exp130 を追加。

## 次アクション

Kaggle train v1 は OOM で失敗。次に再実行する場合は、control 無効化済み package で addonly variant のみを対象にし、GPU コストを明示してユーザー承認を得てから push する。

## Kaggle train push

- 2026-06-26: `make push-kaggle-train EXP=exp130_pfbeam_normalized_diagnostic_score` を実行。
- Kaggle API には到達したが、weekly GPU quota 上限で実行拒否された。
  - message: `Maximum weekly GPU quota of 45.00 hours reached.`
  - status: Kaggle train 未実行
- 同じ kernel id の存在確認:
  - `kaggle kernels pull kentookumura/exp130-pfbn-score-train -p /tmp/kaggle-pull/exp130-pfbn-score-train -m`: `GetKernel` 500。
  - `kaggle kernels logs kentookumura/exp130-pfbn-score-train`: `ListKernelSessionOutput` 404。
  - `kaggle kernels output kentookumura/exp130-pfbn-score-train -p /tmp/kaggle-output/exp130_pfbeam_normalized_diagnostic_score/train_check`: `ListKernelSessionOutput` 404。
- `logs` / `output` が 404 のため、実行セッションは作成されていない扱い。別 slug での再 push はしていない。

## Kaggle train push retry

- 2026-06-27: ユーザーから GPU quota 回復の連絡を受け、Kaggle train push を再試行。
- canonical kernel id `kentookumura/exp130-pfbn-score-train` は `Notebook not found` で新規作成できなかった。
  - `kaggle kernels pull kentookumura/exp130-pfbn-score-train -p /tmp/kaggle-pull/exp130-pfbn-score-train-check -m`: `GetKernel` 500。
  - `kaggle kernels list --mine --search exp130-pfbn-score-train`: Not found。
  - source なし / `run_on_push=false` でも `Notebook not found`。
  - slug-only id は CLI validation で不可。
- Kaggle 側に canonical kernel が存在しないことを確認したうえで、重複回避より実行を優先し、代替 slug に切り替えた。
  - kernel id: `kentookumura/exp130-pfbn-normalized-score-train`
  - title: `exp130 pfbn normalized score train`
  - URL: `https://www.kaggle.com/code/kentookumura/exp130-pfbn-normalized-score-train`
- `kaggle kernels push -p experiments/exp130_pfbeam_normalized_diagnostic_score/kaggle/train`: `Kernel version 1 successfully pushed.`
- `kaggle kernels pull kentookumura/exp130-pfbn-normalized-score-train -p /tmp/kaggle-pull/exp130-pfbn-normalized-score-train-v1 -m`: PASS。
- `kaggle kernels status kentookumura/exp130-pfbn-normalized-score-train`: `KernelWorkerStatus.RUNNING`。
- `timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp130-pfbn-normalized-score-train` は log stream が空のまま継続していたが、ユーザー依頼により監視を停止した。

## Kaggle train v1 failure

- 2026-06-27: `kentookumura/exp130-pfbn-normalized-score-train` は失敗。
- Kaggle UI message: `Your notebook tried to allocate more memory than is available.`。ログ上は `nbclient.exceptions.DeadKernelError: Kernel died` で、Python 例外ではなく OOM による kernel death と判断する。
- 失敗した Kaggle package は古い `config.yaml` のままで、`exp092_full_row_control` と `pfbeam_normalized_diagnostic_addonly` の 2 variant が有効だった。
  - 実行ログ上の active variants: `['exp092_full_row_control', 'pfbeam_normalized_diagnostic_addonly']`
  - control 3 config x 5 folds を先に学習した後、addonly の `lgb2` fold0 完了後付近で OOM。
- 生成物は incomplete。
  - `kaggle/output/train_v1_failed/artifacts/exp130_pfbeam_normalized_diagnostic_score_diagnostic_feature_summary.csv`
  - control の一部 model file のみ。
  - addonly の完走 metrics / model manifest は無し。
- 途中ログから見た `pfbeam_normalized_diagnostic_addonly` の有効性は否定的または不明。
  - `lgb0` は 5 folds 完了。単純平均 RMSE は control 9.499024、addonly 9.563375、差分 +0.064351 で悪化。
  - `lgb1` は 5 folds 完了。単純平均 RMSE は control 9.291006、addonly 9.336510、差分 +0.045503 で悪化。
  - `lgb2` は fold0 のみ完了。control 8.380249、addonly 8.373010、差分 -0.007238 で微改善だが 1 fold のみで判断不能。
  - 完走 metrics / pooled RMSE / ensemble RMSE が無いため最終判定ではないが、途中結果だけでは addonly が有効だったとは言えない。
- 2026-06-27: 正本 `config.yaml` と Kaggle package を揃えるため、train package を再生成。
  - command: `make prepare-kaggle-notebooks EXP=exp130_pfbeam_normalized_diagnostic_score EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp130-pfbn-normalized-score-train --title 'exp130 pfbn normalized score train' --run-on-push --strict"`
  - 再生成後の `kaggle/train/config.yaml` は `exp092_full_row_control.enabled: false`。
- 2026-06-27: ユーザー判断により棄却。途中比較でも `lgb0` / `lgb1` は悪化しており、追加 GPU を使った full rerun は行わない。
- inference port、submit、後続の `u_state_pf_candidate` 入力化は行わない。
