# exp006_hard_well_router_diagnostic セッションノート

## 目的

exp002 / exp003 / exp005 の OOF well-level 差分から hard well と public-like well をタグ付けし、次の all-GR / no-GR / guarded router 実装に使う診断 artifact を作る。

## 現在の状態

- 状態: Kaggle full train 完了、output 取得済み
- 親実験: `exp005_gr_gate_recalibration`
- 新規提出: なし
- exp006 Kaggle train: complete, version 1

## コマンドログ

- 2026-06-01: `uv run python scripts/new_steering.py --experiment exp006_hard_well_router_diagnostic` で steering docs を作成。
- 2026-06-01: `uv run python scripts/new_experiment.py --name exp006_hard_well_router_diagnostic --source experiments/exp005_gr_gate_recalibration` で exp005 から実験を作成。
- 2026-06-01: notebook ファイル名と `settings.py` を exp006 に更新。
- 2026-06-01: `diagnostics.py` を追加し、train notebook の OOF scoring 後に router diagnostic artifact を出力するよう更新。
- 2026-06-01: `uv run python experiments/exp006_hard_well_router_diagnostic/diagnostics.py` を実行し、保存済み exp003/exp005 artifacts から診断 CSV を生成。
- 2026-06-01: `uv run python scripts/validate_experiment.py --experiment exp006_hard_well_router_diagnostic` が通過。
- 2026-06-01: `uv run ruff check experiments/exp006_hard_well_router_diagnostic/diagnostics.py experiments/exp006_hard_well_router_diagnostic/baseline.py experiments/exp006_hard_well_router_diagnostic/settings.py` が通過。
- 2026-06-01: `python -m py_compile experiments/exp006_hard_well_router_diagnostic/diagnostics.py experiments/exp006_hard_well_router_diagnostic/baseline.py experiments/exp006_hard_well_router_diagnostic/settings.py` が通過。
- 2026-06-01: `python -m json.tool` で train / inference notebook の JSON 検査が通過。
- 2026-06-01: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp006_hard_well_router_diagnostic --notebook train --strict` が通過。
- 2026-06-01: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp006_hard_well_router_diagnostic --notebook train --run-on-push --title "exp006 hard well router diagnostic train" --strict` で train package を再生成。
- 2026-06-01: `kaggle kernels push -p experiments/exp006_hard_well_router_diagnostic/kaggle/train` で version 1 を push。URL: https://www.kaggle.com/code/kentookumura/exp006-hard-well-router-diagnostic-train
- 2026-06-01: 5 分間隔で `kaggle kernels status kentookumura/exp006-hard-well-router-diagnostic-train` を監視し、`KernelWorkerStatus.COMPLETE` を確認。
- 2026-06-01: `kaggle kernels output kentookumura/exp006-hard-well-router-diagnostic-train -p /tmp/kaggle-output/exp006_hard_well_router_diagnostic/train` で output と log を取得。
- 2026-06-01: Kaggle output の `metrics.json`、`ablation_metrics.csv`、`well_metrics.csv`、`fold_metrics.csv`、`fold_model_training.csv`、router diagnostic artifacts、train log を `experiments/exp006_hard_well_router_diagnostic/` に反映。

## 変更点

- `diagnostics.py`
  - `well_metrics.csv` を variant ごとに pivot。
  - exp002 all-GR、exp003 no-GR、exp004 any gate、exp005 strict gate の well-level RMSE 差分を作成。
  - `hard_no_gr_candidate`、`public_like_keep_all_gr`、`ambiguous` をタグ付け。
  - inference-safe rule 候補を同一 OOF rows で評価。
- `exp006_hard_well_router_diagnostic_train.ipynb`
  - 既存 exp005 CV ループの後に diagnostic CSV / JSON 出力を追加。
- `exp006_hard_well_router_diagnostic_inference.ipynb`
  - 診断専用のため submission 生成を明示的に無効化。
- Kaggle packaging
  - train notebook の support files 同梱と metadata strict check を確認済み。

## 生成物

- `artifacts/router_diagnostic_well_tags.csv`
- `artifacts/router_condition_summary.csv`
- `artifacts/router_candidate_rules.csv`
- `artifacts/router_diagnostic_metrics.json`

## 結果

| Metric | Value |
| --- | ---: |
| Source wells | 773 |
| Evaluation rows | 3,783,989 |
| exp002 all-GR CV | 14.124569 |
| exp003 no-GR CV | 13.882944 |
| exp004 any low-GR gate CV | 13.932968 |
| exp005 strict low-GR gate CV | 13.936732 |
| Mean fold RMSE, selected | 13.913383 |
| Oracle all-GR/no-GR CV | 13.299351 |

| Bucket | Wells |
| --- | ---: |
| ambiguous | 332 |
| hard_no_gr_candidate | 248 |
| public_like_keep_all_gr | 193 |

最良の inference-safe diagnostic rule は `low_gr_any_to_no_gr` で CV 13.932968。これは exp004 selected gate と同等。Oracle は 13.299351 まで下がるため、router headroom はあるが、現行 low-GR rule だけでは no-GR hurt rate が高い。

## 次のアクション

1. `router_diagnostic_well_tags.csv` を使って `public_like_keep_all_gr` を守る guard 条件を固定する。
2. `hard_no_gr_candidate` を拾う rule-based router を exp007 として fold-safe に実装する。
3. router 自体の target leakage を避けるため、学習済み rule / classifier の特徴量は `condition_*` に限定する。
