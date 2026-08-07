# exp008_gr_ncc_matcher セッションノート

## 目的

バックログ先頭の typewell と horizontal well の GR シグネチャ local matcher / multi-scale NCC を、既存 residual model への add-only 特徴として検証できる形に実装する。

## 現在の状態

- 状態: Kaggle full CV 完了、提出なし
- 親実験: `exp007_hard_well_router`
- selected candidate: `gr_ncc_no_gr_multi`
- CV: 14.641514
- LB: 未提出

## コマンドログ

- 2026-06-01: `uv run python scripts/new_steering.py --experiment exp008_gr_ncc_matcher` で steering docs を作成。
- 2026-06-01: `uv run python scripts/new_experiment.py --name exp008_gr_ncc_matcher --source experiments/exp007_hard_well_router` で exp007 から実験を作成。
- 2026-06-01: train / inference notebook は単体 ablation runner に戻すため exp003 の notebook をベースに差し替え、exp008 名に更新。
- 2026-06-01: `baseline.py` に paired typewell 読み込み helper、GR NCC feature set、multi-scale NCC search、typewell GR interpolation 特徴を追加。
- 2026-06-01: `config.yaml` を exp008 用に更新し、`control_exp002_all`、`control_exp003_no_gr`、`gr_ncc_no_gr_multi`、`gr_ncc_all_multi` を定義。
- 2026-06-01: train / inference notebook の feature frame 作成時に `read_typewell_for_horizontal_path(path)` を渡すよう更新。
- 2026-06-01: 1 train well の sanity check で `gr_ncc_no_gr_multi` feature frame 生成を確認。例: `gr_ncc_best_score=0.418298`、`gr_ncc_best_shift_tvt=20.0`。
- 2026-06-01: `uv run python scripts/validate_experiment.py --experiment exp008_gr_ncc_matcher` が通過。
- 2026-06-01: `uv run ruff check experiments/exp008_gr_ncc_matcher/baseline.py experiments/exp008_gr_ncc_matcher/settings.py` が通過。
- 2026-06-01: `python3 -m py_compile experiments/exp008_gr_ncc_matcher/baseline.py experiments/exp008_gr_ncc_matcher/settings.py` が通過。
- 2026-06-01: train / inference notebook の JSON 検査が通過。
- 2026-06-01: `uv run pytest` が通過。9 tests passed。
- 2026-06-01: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp008_gr_ncc_matcher --notebook train --run-on-push --title "exp008 gr ncc matcher train" --strict` が通過。
- 2026-06-01: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp008_gr_ncc_matcher --notebook inference --run-on-push --title "exp008 gr ncc matcher inference" --strict` が通過。
- 2026-06-01: `uv run python scripts/record_experiment.py --experiment exp008_gr_ncc_matcher --status scaffold_completed ...` で `metrics.json` と `experiment_summary.md` を更新。
- 2026-06-01: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp008_gr_ncc_matcher --notebook train --run-on-push --title "exp008 gr ncc matcher train" --strict` で train package を再生成。
- 2026-06-01: `kaggle kernels push -p experiments/exp008_gr_ncc_matcher/kaggle/train` で version 1 を push。URL: https://www.kaggle.com/code/kentookumura/exp008-gr-ncc-matcher-train
- 2026-06-01: `kaggle kernels status kentookumura/exp008-gr-ncc-matcher-train` を監視し、`KernelWorkerStatus.COMPLETE` を確認。
- 2026-06-01: `kaggle kernels output kentookumura/exp008-gr-ncc-matcher-train -p /tmp/kaggle-output/exp008_gr_ncc_matcher/train` で output と kernel log を取得。
- 2026-06-01: Kaggle output の `metrics.json`、`artifacts/ablation_metrics.csv`、`fold_metrics.csv`、`fold_model_training.csv`、`well_metrics.csv`、train log を `experiments/exp008_gr_ncc_matcher/` に反映。

## 変更点

- `gr_ncc_no_gr_multi`: exp003 の `no_gr_signal` feature set に NCC 特徴を追加。
- `gr_ncc_all_multi`: exp002 の raw-GR feature set に NCC 特徴を追加。
- NCC 候補:
  - `w25_r150`: smoothing window 25、typewell TVT shift radius 150、step 10。
  - `w75_r300`: smoothing window 75、typewell TVT shift radius 300、step 20。
- NCC search は known-prefix recent TVT slope を prior とし、bounded shift と reverse direction を許す。

## リーク対策

- 同一 well は fold 間で分割しない。
- GR NCC は horizontal GR、typewell TVT/GR、MD、known `TVT_input` prefix のみから作る。
- hidden / validation evaluation-zone true `TVT` は feature alignment に使わない。
- train-only formation columns は使わない。

## 結果

| Variant | Feature Set | CV | exp002 差分 |
| --- | --- | ---: | ---: |
| `control_exp003_no_gr` | `no_gr_signal` | 13.882944 | -0.241625 |
| `control_exp002_all` | `all` | 14.124569 | 0.000000 |
| `gr_ncc_no_gr_multi` | `no_gr_signal_plus_gr_ncc` | 14.641514 | +0.516945 |
| `gr_ncc_all_multi` | `all_plus_gr_ncc` | 14.661017 | +0.536448 |

selected `gr_ncc_no_gr_multi` は CV 14.641514 で、`control_exp003_no_gr` より 0.758570 悪く、`control_exp002_all` より 0.516945 悪い。NCC features は現行設計では採用しない。

## 次のアクション

1. exp008 は提出しない。
2. 次は backlog 先頭の formation columns を直接使わない structural guide に進む。
