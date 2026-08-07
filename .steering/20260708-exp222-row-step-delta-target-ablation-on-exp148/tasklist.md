# タスクリスト

## 進行中

- なし

## ブロック中

- なし

## 完了

- `.steering/20260708-exp222-row-step-delta-target-ablation-on-exp148/` を作成した。
- `experiments/exp222_row_step_delta_target_ablation_on_exp148/` を作成した。
- exp148 helper を元に `row_step_delta_target_ablation_on_exp148.py` を追加した。
- `target_anchor_delta`、`target_tvt`、`target_step_delta`、`row_number`、`row_within_tail` の生成を実装した。
- OOF prediction を well-wise cumulative sum で `pred_tvt` に復元する評価を実装した。
- cumulative drift readout を `exp222_row_step_delta_target_ablation_on_exp148_cumulative_drift.csv` として保存するようにした。
- train notebook source `exp222_row_step_delta_target_ablation_on_exp148_train.py` を追加した。
- `config.yaml` に CPU only、`lgb_config_indices: [0]`、5 folds、control 再学習なしを記録した。
- `docs/06_reproducibility.md` に沿って seed / CPU flags / SHA 記録方針を design と config に記録した。
- Jupytext で train `.py` を `.ipynb` に変換し、`--test` を通した。
- `py_compile`、`ruff --select F821`、`validate-exp` を通した。
- parent compact notebook と exp222 notebook/helper の章立て・行数比較を `SESSION_NOTES.md` に記録した。
- Kaggle train package を CPU / lgb0 / strict で準備し、metadata と bootstrap config を確認した。
- 初回の長い slug push は `SaveKernel` 400 で失敗したため、同じ exp のまま短縮 slug `kentookumura/exp222-row-step-delta-target-train` へ再 prepare した。
- 短縮 slug push は CPU session 上限で実行開始できなかった。
- CPU slot が空いた後、`kentookumura/exp222-row-step-delta-target-train` への retry は `Notebook not found` で失敗した。
- Source kernel `exp072` / `exp145` が pull できることを確認し、同じ exp のまま新短縮 slug `kentookumura/exp222-stepdelta-lgb0` に切り替えた。
- `kentookumura/exp222-stepdelta-lgb0` v1 push に成功し、status RUNNING を確認した。
- `kentookumura/exp222-stepdelta-lgb0` v1 は LightGBM 学習開始前に `DeadKernelError: Kernel died` で失敗した。
- v1 失敗原因として、入力確認セルが exp145 learned likelihood cache 378万行を preview 目的で全量ロードし、本処理でも再ロードしていた点を確認した。
- v2 修正として入力確認を header + `nrows=8` preview に変更し、learned feature join のキー順一致 fast path と一時 DataFrame 破棄を追加した。
- v2 修正後の `py_compile`、`ruff --select F821`、Jupytext `--test`、`make validate-exp` を通した。
- `kentookumura/exp222-stepdelta-lgb0` v2 push に成功し、status RUNNING を確認した。
- `kentookumura/exp222-stepdelta-lgb0` v2 も LightGBM 学習開始前に `DeadKernelError: Kernel died` で失敗した。
- v3 修正として column-wise finite check、anchor map、step target の小型 order frame、単一 training surface 時の feature matrix 作成後 feature-column drop、fold 後の LightGBM 一時 object 破棄、stage log を追加した。
- v3 修正後の `py_compile`、`ruff --select F821`、Jupytext `--test`、`make validate-exp` を通した。
- `kentookumura/exp222-stepdelta-lgb0` v3 push に成功し、status RUNNING を確認した。
- `kentookumura/exp222-stepdelta-lgb0` v3 が COMPLETE。pooled TVT RMSE 15.301575 で exp148 lgb0 8.599786 から +6.701789 悪化した。
- distance 1000_plus bucket が 16.933071、worst-well `1b1eba53` が 67.727455、cumulative drift final error -69.507812 となり、step-delta 累積復元の失敗を確認した。
- train-side rejected とし、lgb1/lgb2 展開、inference port、submit は行わない方針にした。

## 残り

- なし
