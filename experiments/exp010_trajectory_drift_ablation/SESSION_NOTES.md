# exp010_trajectory_drift_ablation セッションノート

## 目的

formation guide なしで、trajectory 形状から TVT drift residual を補正できるかを ablation する。

## 現在の状態

- 状態: Kaggle full CV 完了、提出なし
- 親実験: `exp009_formation_surface_guide`
- selected candidate: `trajectory_full_no_gr`
- CV: `trajectory_full_no_gr` 14.236694
- LB: 未提出

## コマンドログ

- 2026-06-02: `task new-steering EXP=exp010_trajectory_drift_ablation` は `task` 未インストールで失敗。
- 2026-06-02: `uv run python scripts/new_steering.py --experiment exp010_trajectory_drift_ablation` で steering docs を作成。
- 2026-06-02: `uv run python scripts/new_experiment.py --name exp010_trajectory_drift_ablation --source experiments/exp009_formation_surface_guide` で exp009 から実験を作成。
- 2026-06-02: train / inference notebook を exp010 名にリネーム。
- 2026-06-02: `config.yaml` を exp010 用に更新し、trajectory direction / slope / full variants を定義。
- 2026-06-02: `baseline.py` に inference-safe trajectory drift feature groups を追加。
- 2026-06-02: 1 train well で `trajectory_full_no_gr` の feature frame sanity check を実行。`active_features=60`、new interaction columns の生成を確認。
- 2026-06-02: `python3 -m py_compile experiments/exp010_trajectory_drift_ablation/baseline.py experiments/exp010_trajectory_drift_ablation/settings.py` が通過。
- 2026-06-02: `uv run ruff check experiments/exp010_trajectory_drift_ablation/baseline.py experiments/exp010_trajectory_drift_ablation/settings.py` が通過。
- 2026-06-02: `uv run python scripts/validate_experiment.py --experiment exp010_trajectory_drift_ablation` が通過。
- 2026-06-02: `uv run pytest` が通過。9 tests passed。
- 2026-06-02: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp010_trajectory_drift_ablation --notebook train --run-on-push --title "exp010 trajectory drift ablation train" --strict` が通過。
- 2026-06-02: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp010_trajectory_drift_ablation --notebook inference --run-on-push --title "exp010 trajectory drift ablation inference" --strict` が通過。
- 2026-06-02: `uv run python scripts/update_experiment_summary.py` で `experiment_summary.md` に exp010 scaffold row を追加。
- 2026-06-02: `kaggle kernels push -p experiments/exp010_trajectory_drift_ablation/kaggle/train` で version 1 を push。URL: https://www.kaggle.com/code/kentookumura/exp010-trajectory-drift-ablation-train
- 2026-06-02: `kaggle kernels status kentookumura/exp010-trajectory-drift-ablation-train` を監視し、`KernelWorkerStatus.COMPLETE` を確認。
- 2026-06-02: `kaggle kernels output kentookumura/exp010-trajectory-drift-ablation-train -p /tmp/kaggle-output/exp010_trajectory_drift_ablation/train` で output と kernel log を取得。
- 2026-06-02: Kaggle output の `metrics.json`、`artifacts/ablation_metrics.csv`、`fold_metrics.csv`、`fold_model_training.csv`、`well_metrics.csv`、train log を `experiments/exp010_trajectory_drift_ablation/` に反映。
- 2026-06-03: `uv run python scripts/new_steering.py --experiment trajectory_feature_error_audit --title 'trajectory feature error audit'` で診断用 steering docs を作成。
- 2026-06-03: `uv run python studies/trajectory_feature_error_audit.py` で `well_metrics.csv` と exp006 router tags を結合し、well-level trajectory feature error audit を実行。
- 2026-06-03: 診断 output を `artifacts/trajectory_feature_error_audit/` に保存。`trajectory_audit_report.md`、`trajectory_audit_metrics.json`、well deltas / group summary / top hurt-help CSV を生成。

## 変更点

- `trajectory_direction_no_gr`: exp003 の `no_gr_signal` に azimuth / signed direction / final-axis projection を追加。
- `trajectory_slope_no_gr`: exp003 の `no_gr_signal` に inclination、`dZ/dMD`、`dXY/dMD`、prefix trajectory slope を追加。
- `trajectory_full_no_gr`: direction、slope、recent TVT slope との interaction をまとめて追加。
- `trajectory_full_all`: exp002 の all-GR feature set に同じ trajectory full group を追加。

## リーク対策

- 同一 well は fold 間で分割しない。
- `TVT_input` の既知 prefix 以外から target-derived feature を作らない。
- train-only formation columns は使わない。
- GR NCC と formation guide は disabled のままにする。

## 結果

| Variant | Feature Set | CV | exp002 差分 |
| --- | --- | ---: | ---: |
| `control_exp003_no_gr` | `no_gr_signal` | 13.882944 | -0.241625 |
| `trajectory_direction_no_gr` | `no_gr_signal_plus_trajectory_direction` | 14.023223 | -0.101346 |
| `control_exp002_all` | `all` | 14.124569 | 0.000000 |
| `trajectory_slope_no_gr` | `no_gr_signal_plus_trajectory_slope` | 14.177009 | +0.052440 |
| `trajectory_full_no_gr` | `no_gr_signal_plus_trajectory_drift` | 14.236694 | +0.112125 |
| `trajectory_full_all` | `all_plus_trajectory_drift` | 14.332077 | +0.207508 |

selected `trajectory_full_no_gr` は CV 14.236694。best は `control_exp003_no_gr` 13.882944 で、trajectory add-only variants はすべて exp003 no-GR control より悪化した。提出しない。

## 追加診断: trajectory_feature_error_audit

`trajectory_full_no_gr` は 773 wells / 3,783,989 eval rows で、`control_exp003_no_gr` 13.882944 から 14.236694 へ悪化。well 単位では 351 wells が meaningful hurt、308 wells が meaningful better。

悪化が強い条件:

| 条件 | wells | exp003 CV | trajectory_full CV | 差分 | hurt rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| hard_no_gr_candidate | 248 | 16.721940 | 17.688955 | +0.967015 | 0.592742 |
| fold 4 | 154 | 15.648648 | 16.556048 | +0.907400 | 0.487013 |
| steep_trajectory | 186 | 15.208859 | 15.979471 | +0.770612 | 0.510753 |
| gr_weak_any | 297 | 13.180229 | 13.895116 | +0.714887 | 0.444444 |
| high_gr_missing | 293 | 13.250584 | 13.959989 | +0.709405 | 0.443686 |
| long_eval | 235 | 14.240710 | 14.886789 | +0.646079 | 0.451064 |

一方で `public_like_keep_all_gr` 193 wells では `trajectory_full_no_gr` が 14.684601 から 14.289867 へ改善しており、trajectory geometry は全面禁止ではなく router / selector の候補にする。

結論: full trajectory add-only は採用しない。次の tracker divergence features では、hard-no-GR / steep trajectory / high GR missing / long eval を事前診断し、単純な追加特徴ではなく confidence / selector / divergence として扱う。

## 次のアクション

1. exp010 は提出しない。
2. `experiment_summary.md` と `backlog/KAGGLE_DIRECTION.md` を更新する。
3. 次候補は `exp011_tracker_divergence_features`。ただし trajectory full を直接再投入せず、audit の悪化条件を selector と runtime / confidence 設計に使う。
