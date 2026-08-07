# exp168_gr_matching_pair_visualization セッションノート

## 目的

GR マッチングでどの水平井 GR window と typewell GR window が比較・採用されたかを、
Kaggle Notebook output の PNG / HTML / CSV として確認できるようにする。

## 現在の状態

- Route: pf_beam
- 状態: completed_visualization_diagnostic
- CV: まだなし
- LB: まだなし

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

```bash
uv run python scripts/new_steering.py --experiment exp168_gr_matching_pair_visualization
uv run python scripts/new_experiment.py --name exp168_gr_matching_pair_visualization --source templates/experiment
.venv/bin/python -m py_compile experiments/exp168_gr_matching_pair_visualization/exp168_gr_matching_pair_visualization_train.py experiments/exp168_gr_matching_pair_visualization/exp168_gr_matching_pair_visualization_inference.py
.venv/bin/ruff check experiments/exp168_gr_matching_pair_visualization/exp168_gr_matching_pair_visualization_train.py experiments/exp168_gr_matching_pair_visualization/exp168_gr_matching_pair_visualization_inference.py --select F821,F722,F823
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp168_gr_matching_pair_visualization/exp168_gr_matching_pair_visualization_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp168_gr_matching_pair_visualization/exp168_gr_matching_pair_visualization_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp168_gr_matching_pair_visualization/exp168_gr_matching_pair_visualization_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp168_gr_matching_pair_visualization/exp168_gr_matching_pair_visualization_inference.py
uv run python scripts/validate_experiment.py --experiment exp168_gr_matching_pair_visualization
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp168_gr_matching_pair_visualization --notebook train --kernel-id kentookumura/exp168-gr-matching-pair-visualization-train --title 'exp168 gr matching pair visualization train' --run-on-push --strict
make push-kaggle-train EXP=exp168_gr_matching_pair_visualization
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 experiments/exp168_gr_matching_pair_visualization/exp168_gr_matching_pair_visualization_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 experiments/exp168_gr_matching_pair_visualization/exp168_gr_matching_pair_visualization_inference.py
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp168_gr_matching_pair_visualization --notebook train --kernel-id kentookumura/exp168-gr-matching-pair-visualization-train --title 'exp168 gr matching pair visualization train' --run-on-push --strict
make push-kaggle-train EXP=exp168_gr_matching_pair_visualization
kaggle kernels status kentookumura/exp168-gr-matching-pair-visualization-train
kaggle kernels logs kentookumura/exp168-gr-matching-pair-visualization-train
kaggle kernels output kentookumura/exp168-gr-matching-pair-visualization-train -p experiments/exp168_gr_matching_pair_visualization/kaggle/output/train_v2
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 experiments/exp168_gr_matching_pair_visualization/exp168_gr_matching_pair_visualization_train.py
.venv/bin/python -m py_compile experiments/exp168_gr_matching_pair_visualization/exp168_gr_matching_pair_visualization_train.py
.venv/bin/ruff check experiments/exp168_gr_matching_pair_visualization/exp168_gr_matching_pair_visualization_train.py --select F821,F722,F823
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp168_gr_matching_pair_visualization/exp168_gr_matching_pair_visualization_train.py
uv run python scripts/validate_experiment.py --experiment exp168_gr_matching_pair_visualization
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp168_gr_matching_pair_visualization --notebook train --kernel-id kentookumura/exp168-gr-matching-pair-visualization-train --title 'exp168 gr matching pair visualization train' --run-on-push --strict
make push-kaggle-train EXP=exp168_gr_matching_pair_visualization
kaggle kernels status kentookumura/exp168-gr-matching-pair-visualization-train
kaggle kernels logs kentookumura/exp168-gr-matching-pair-visualization-train
kaggle kernels output kentookumura/exp168-gr-matching-pair-visualization-train -p experiments/exp168_gr_matching_pair_visualization/kaggle/output/train_v5
```

## Kaggle 実行結果

- Kernel: `kentookumura/exp168-gr-matching-pair-visualization-train`
- URL: https://www.kaggle.com/code/kentookumura/exp168-gr-matching-pair-visualization-train
- v1: `ERROR`。Papermill が `ValueError: No kernel name found in notebook and no override provided.` で起動前に失敗。
- 修正: Jupytext で notebook metadata に `kernelspec.name=python3` を追加し、同じ canonical kernel id に v2 として再 push。
- v2: `COMPLETE`。
- v3: `COMPLETE`。query vs matched の単純 overlay と exp098 lgb1 OOF error good/bad HTML を追加。
- v5: `COMPLETE`。top-k local minima、true-near minimum、shift-cost curve、全体 GR context、wrong-depth bucket 別 OOF 集計を追加。
- 実行時間: notebook summary 上 v2 `35.551281` 秒、v3 `79.275075` 秒、v5 `171.863769` 秒。
- rows scored: 4096。
- wells scored: 16。
- figures: v5 は 48 global/local diagnostics、32 four-panel、32 simple overlay。
- filters: `raw`, `rolling_median_11`, `savgol_31_p2`, `fft_notch_top2`。
- eval regions: `hidden_tail`, `prefix_backtest`。
- output 取得先 v2: `experiments/exp168_gr_matching_pair_visualization/kaggle/output/train_v2/`。
- output 取得先 v3: `experiments/exp168_gr_matching_pair_visualization/kaggle/output/train_v3/`。
- output 取得先 v5: `experiments/exp168_gr_matching_pair_visualization/kaggle/output/train_v5/`。
- v3 OOF join: exp098 lgb1 OOF predictions 15,135,956 rows を scan し、selected pair 9 unique IDs に `oof_abs_error` を付与。
- v5 OOF join: exp098 lgb1 OOF predictions 15,135,956 rows を scan し、scored/selected pair 512 unique IDs に `oof_abs_error` を付与。
- v3 additional outputs:
  - `artifacts/exp168_gr_matching_pair_visualization_good_bad_oof_index.html`
  - `artifacts/simple_overlay/*.png` 32 files
- v5 additional outputs:
  - `artifacts/exp168_gr_matching_pair_visualization_wrong_depth_index.html`
  - `artifacts/exp168_gr_matching_pair_visualization_global_local_index.html`
  - `artifacts/exp168_gr_matching_pair_visualization_oof_bucket_summary.csv`
  - `artifacts/global_local/*.png` 48 files

## 変更点

- `config.yaml` を pf_beam route の visualization diagnostic 用に更新。
- active variant 数: 0。
- LightGBM config 数: 0。
- fold 数: 0。
- 合計 booster 数: 0。
- 親実験 control 再学習: なし。
- train notebook で raw train input から scored pair CSV、selected pair CSV、PNG、HTML index、summary JSON を作る。
- v5 では best だけではなく、local minimum の top-k、true-near minimum、second/third candidate、best-true が 6/10/15 ft 以上ずれる bucket を出す。
- v5 では query GR と best/alternative candidate GR の重ね描きに加え、水平井全体 GR と typewell 全体 GR 上の候補位置も同じ PNG に出す。
- inference notebook は提出を作らない診断専用 stub。

## 再現性メモ

- seed policy: no RNG。row sampling と example selection は deterministic。
- stochastic components: なし。
- CPU/GPU runtime: CPU のみ。GPU 学習なし。
- Kaggle kernel id / version: `kentookumura/exp168-gr-matching-pair-visualization-train` v5。
- input / feature schema SHA: `artifacts/exp168_gr_matching_pair_visualization_summary.json` と `input_summary.csv` に記録。
- feature content SHA: `scored_pairs.csv.gz` decompressed SHA `0e86bbf8b3433acf18a72bbe11950626e2425c8ab863cf1784218c17da4af69a`。
- selected pairs SHA: `51e3a74317a7980874c1329c8c23a3a8b36836fe5dd4194b155157eceb795c40`。
- wrong-depth index SHA: `d1bf822853f50e039cdb78cb59c4fecd2d35a766456bb2f41ba42b692de4f866`。
- OOF bucket summary SHA: `5b553761f42d45d833b1fd10b1770e76e6433c701ad49f534435619814e66d42`。
- model manifest / model SHA: 対象外。
- prediction SHA: 対象外。
- submission SHA: 対象外。
- rerun check: 未実行。

## 次のアクション

1. `experiments/exp168_gr_matching_pair_visualization/kaggle/output/train_v5/artifacts/exp168_gr_matching_pair_visualization_wrong_depth_index.html` で wrong-depth bucket、OOF abs error、red best / cyan true-near / gray alternatives を確認する。
2. true-near delta cost が小さい wrong-depth 例は top-k/posterior 化の候補、delta cost が大きい例は GR 尤度以外の prior / likelihood 補正候補として読む。
3. 必要なら `config.yaml` の `audit.max_wells` / `well_include` / `max_total_figures` を調整して、対象 well を増やした追加実行を同じ kernel id に積む。
