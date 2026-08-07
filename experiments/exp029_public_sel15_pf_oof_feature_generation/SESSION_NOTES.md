# exp029_public_sel15_pf_oof_feature_generation セッションノート

## 目的

公開 replay anchor `exp027_public_replay_needless090_sel15_spread3` の sel15 PF/Beam を、後続の selector / meta-stack が学習に使える train-side OOF-like feature artifact に変換する。

## 現在の状態

- Route: pf_beam
- 状態: all-well cutoff 0.65 artifact 完了
- CV: まだなし
- LB: まだなし
- 親実験: `exp027_public_replay_needless090_sel15_spread3`
- smoke run: 20 wells / cutoff 0.65 / 16 seeds / 250 particles
- current run config: all wells / cutoff 0.65 / 16 seeds / 250 particles / gzip output
- outputs:
  - `features/public_sel15_pf_oof_features.csv`
  - `artifacts/public_sel15_pf_oof_well_summary.csv`

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

- 2026-06-07: `uv run python scripts/new_steering.py --experiment exp029_public_sel15_pf_oof_feature_generation` で steering docs を作成。
- 2026-06-07: `uv run python scripts/new_experiment.py --name exp029_public_sel15_pf_oof_feature_generation` で実験フォルダを作成。
- 2026-06-07: `.steering/20260607-exp029-public-sel15-pf-oof-feature-generation/` に要件、設計、タスクを記録。
- 2026-06-07: `public_sel15_pf_oof.py` を追加し、train well の途中以降を隠す cutoff、PF likelihood ensemble、14-config beam ensemble、public selector、feature CSV 追記保存を実装。
- 2026-06-07: `config.yaml`、README、train/inference notebook、result、metrics を exp029 用に更新。
- 2026-06-07: `uv run python -m py_compile experiments/exp029_public_sel15_pf_oof_feature_generation/public_sel15_pf_oof.py experiments/exp029_public_sel15_pf_oof_feature_generation/settings.py` が通過。
- 2026-06-07: `uv run ruff check experiments/exp029_public_sel15_pf_oof_feature_generation/public_sel15_pf_oof.py experiments/exp029_public_sel15_pf_oof_feature_generation/settings.py` が通過。
- 2026-06-07: `uv run python scripts/validate_experiment.py --experiment exp029_public_sel15_pf_oof_feature_generation` が通過。
- 2026-06-07: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp029_public_sel15_pf_oof_feature_generation --notebook train --strict` が通過。
- 2026-06-07: `uv run python experiments/exp029_public_sel15_pf_oof_feature_generation/public_sel15_pf_oof.py --allow-local --debug-n-wells 1 --n-seeds 2 --n-particles 20 --cutoffs 0.8` が通過。`000d7d20` 1 well / 1056 rows、PF RMSE diagnostic 10.433962、last-anchor 11.438746、beam 11.641419。
- 2026-06-07: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp029_public_sel15_pf_oof_feature_generation --notebook train --run-on-push --strict` 後、default long slug `kentookumura/exp029-public-sel15-pf-oof-feature-generation-train` で push したが Kaggle API 400。title/slug mismatch を直しても 400 だったため、未作成の長い slug を捨て、短い canonical kernel `kentookumura/exp029-sel15-pf-oof-train` に切り替えた。
- 2026-06-07: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp029_public_sel15_pf_oof_feature_generation --notebook train --kernel-id kentookumura/exp029-sel15-pf-oof-train --title "exp029 sel15 pf oof train" --run-on-push --strict` が通過。
- 2026-06-07: `kaggle kernels push -p experiments/exp029_public_sel15_pf_oof_feature_generation/kaggle/train` で version 1 push 成功。ただし Kaggle smoke は `distance_bucket()` の 2500+ rows で `IndexError`。原因は `np.where` が両分岐を評価し、`labels[4]` が out of bounds になったこと。
- 2026-06-07: `distance_bucket()` を bucket index clip 方式に修正。`uv run ruff check ...`、`uv run python -m py_compile ...`、`uv run python experiments/exp029_public_sel15_pf_oof_feature_generation/public_sel15_pf_oof.py --allow-local --debug-n-wells 1 --n-seeds 2 --n-particles 20 --cutoffs 0.65` が通過。local smoke は 1847 rows、PF RMSE 8.776856、last-anchor 10.200872、beam 10.532402。
- 2026-06-07: 同じ short kernel id に version 2 を push。`kaggle kernels pull kentookumura/exp029-sel15-pf-oof-train -p /tmp/kaggle-pull/exp029-sel15-pf-oof-train -m` で存在確認。
- 2026-06-07: `kaggle kernels logs kentookumura/exp029-sel15-pf-oof-train` で version 2 完了ログを確認。20 wells / 43,542 rows、PF RMSE diagnostic 10.381203、last-anchor 16.266463、beam 15.154074。
- 2026-06-07: `kaggle kernels output kentookumura/exp029-sel15-pf-oof-train -p /tmp/kaggle-output/exp029_public_sel15_pf_oof_feature_generation/train_v2` で output を取得。feature CSV、well summary、metrics、log を取得し、feature / summary / log を実験フォルダ配下の ignored 生成物にも同期。
- 2026-06-07: ユーザー指示により `all wells / cutoff=0.65 / 16 seeds / 250 particles` へ進行。`config.yaml` の `runtime.debug_n_wells` を `null`、`runtime.output_compression` を `gzip`、`runtime.run_label` を `all_wells_cutoff065_seed16_particles250` に変更。
- 2026-06-07: `public_sel15_pf_oof.py` に gzip output 対応を追加し、feature output を `features/public_sel15_pf_oof_features.csv.gz` に変更。local smoke と gzip 読み込み確認が通過。
- 2026-06-07: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp029_public_sel15_pf_oof_feature_generation --notebook train --kernel-id kentookumura/exp029-sel15-pf-oof-train --title "exp029 sel15 pf oof train" --run-on-push --strict` が通過。
- 2026-06-07: `kaggle kernels push -p experiments/exp029_public_sel15_pf_oof_feature_generation/kaggle/train` で version 3 push 成功。Kaggle queue が長く、しばらく logs/output は空だったが、ユーザー確認後に完了を確認。
- 2026-06-07: `kaggle kernels logs kentookumura/exp029-sel15-pf-oof-train` で version 3 完了ログを確認。773 wells / 1,782,279 rows、PF RMSE diagnostic 15.172636、last-anchor 18.284054、beam 18.122632。
- 2026-06-07: `kaggle kernels output kentookumura/exp029-sel15-pf-oof-train -p /tmp/kaggle-output/exp029_public_sel15_pf_oof_feature_generation/train_v3` で output を取得。`features/public_sel15_pf_oof_features.csv.gz` は 242MB、well summary は 773 wells 分。feature / summary / log を実験フォルダ配下の ignored 生成物にも同期。

### 予定

```bash
task validate-exp EXP=exp029_public_sel15_pf_oof_feature_generation
task prepare-kaggle-notebooks EXP=exp029_public_sel15_pf_oof_feature_generation EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp029-sel15-pf-oof-train --title 'exp029 sel15 pf oof train' --run-on-push --strict"
task push-kaggle-train EXP=exp029_public_sel15_pf_oof_feature_generation
kaggle kernels pull <kernel> -p /tmp/kaggle-pull/<slug> -m
kaggle kernels logs <kernel>
kaggle kernels output <kernel> -p /tmp/kaggle-output/exp029_public_sel15_pf_oof_feature_generation/train
```

## 変更点

- Public notebook の inference-only replay から、train-side feature generation に変更。
- cutoff 以降の `TVT_input` を NaN にした train well の途中以降を隠した疑似 test tail 上で PF/Beam を実行。
- row feature と well summary を保存し、selector/meta-stack の入力にする。
- 参照 OOF path が未設定でも schema を固定するため `exp026_oof` 差分列は NaN で保存する。

## 結果

- Local smoke: PASS
- rows: 1056
- generated feature columns: `pf_pred`, scale 別 PF、`beam_pred`, `beam_spread`, likelihood gap / entropy、selected scale、prefix length、distance bucket、GR availability、last-anchor / optional exp026 OOF 差分。
- Kaggle smoke version 1: FAILED (`distance_bucket` 2500+ bucket `IndexError`)
- Kaggle smoke version 2: PASS
- Kaggle kernel: `kentookumura/exp029-sel15-pf-oof-train` v2
- Kaggle rows: 43,542
- Kaggle smoke diagnostics: PF RMSE 10.381203、last-anchor RMSE 16.266463、beam RMSE 15.154074
- output: `/tmp/kaggle-output/exp029_public_sel15_pf_oof_feature_generation/train_v2`
- All-well cutoff 0.65 version 3: PASS
- All-well kernel: `kentookumura/exp029-sel15-pf-oof-train` v3
- All-well rows: 1,782,279
- All-well diagnostics: PF RMSE 15.172636、last-anchor RMSE 18.284054、beam RMSE 18.122632
- All-well feature artifact: `features/public_sel15_pf_oof_features.csv.gz`、242MB
- All-well output: `/tmp/kaggle-output/exp029_public_sel15_pf_oof_feature_generation/train_v3`
- 注意: `reference_oof_rows=0` なので、`exp026_oof` と `pf_pred_minus_exp026_oof` は schema 上はあるが未接続。

## 次のアクション

1. all-well summary で PF が hold より悪い well の条件を分析する。
2. exp026 OOF / postprocessed OOF と row id を揃えて結合し、`public_sel15_pf_candidate_selector` または `public_sel15_pf_meta_stack` に進む。
