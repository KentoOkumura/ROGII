# exp134_self_gr_multiscale_longtail_gate セッションノート

## 現在の状態

- status: `completed_train_side_rejected_no_submit`
- route: `ml_model`
- parent: `exp090_lateral_self_gr_match_pseudotail_probe`
- diagnostic parents: `exp072`, `exp073`, `exp092`, `exp126`, `exp128`
- LightGBM 学習: なし
- submission: なし

## 実装内容

- `.steering/20260626-exp134-self-gr-multiscale-longtail-gate/` を作成し、requirements / design / tasklist を記入。
- `experiments/exp134_self_gr_multiscale_longtail_gate/` を exp125 から作成。
- `config.yaml` を self-GR multiscale longtail gate posthoc audit 用に更新。
- `self_gr_multiscale_longtail_gate.py` を追加。
  - exp072 full replay feature cache を読む。
  - raw train horizontal wells から exp090 相当の multi-scale self-GR signal を再生成する。
  - `self_gr_sc25_delta_tvt`、`self_gr_sc25_score`、`self_gr_sc25_l2`、scale disagreement、GR missingness、distance / PF-dense disagreement context を作る。
  - `likpf_mean` と `tvt_dense` low-frequency gate variants を posthoc 比較する。
  - metrics、by-well、bucket、common-worst、signal summary、gate prediction sample、feature schema、summary を保存する。
- train notebook を 4 セクション構成に更新。
- inference notebook は no-submission diagnostic summary のみ。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp134_self_gr_multiscale_longtail_gate
uv run python scripts/new_experiment.py --name exp134_self_gr_multiscale_longtail_gate --source experiments/exp125_confidence_gate_continuity_rawtest_parity
```

## 次のアクション

- なし。self-GR multiscale longtail gate は rejected として完了。

## 検証

- `uv run python -m py_compile experiments/exp134_self_gr_multiscale_longtail_gate/self_gr_multiscale_longtail_gate.py experiments/exp134_self_gr_multiscale_longtail_gate/settings.py`: PASS
- `python3 -m json.tool experiments/exp134_self_gr_multiscale_longtail_gate/exp134_self_gr_multiscale_longtail_gate_train.ipynb`: PASS
- `python3 -m json.tool experiments/exp134_self_gr_multiscale_longtail_gate/exp134_self_gr_multiscale_longtail_gate_inference.ipynb`: PASS
- `uv run python scripts/validate_experiment.py --experiment exp134_self_gr_multiscale_longtail_gate`: PASS
- `uv run ruff check experiments/exp134_self_gr_multiscale_longtail_gate/self_gr_multiscale_longtail_gate.py experiments/exp134_self_gr_multiscale_longtail_gate/settings.py`: PASS
- `uv run ruff format --check experiments/exp134_self_gr_multiscale_longtail_gate/self_gr_multiscale_longtail_gate.py experiments/exp134_self_gr_multiscale_longtail_gate/settings.py`: PASS
- `make prepare-kaggle-notebooks EXP=exp134_self_gr_multiscale_longtail_gate EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp134-self-gr-gate-train --title 'exp134 self gr gate train' --run-on-push --strict"`: PASS
- generated train package: `experiments/exp134_self_gr_multiscale_longtail_gate/kaggle/train`
- generated kernel id: `kentookumura/exp134-self-gr-gate-train`
- generated metadata: GPU disabled, internet disabled, run_on_push true, competition source `rogii-wellbore-geology-prediction`, kernel sources `exp072`, `exp073`, `exp092`
- bootstrap manifest includes `config.yaml` SHA `c772619d71df3b8638c05c4cca8e77a4688e2456681aab1dcd160bf038567c92` and `self_gr_multiscale_longtail_gate.py` SHA `06890115c4c7ad194026d16580dbd116037d2763dfc7887c4bf79bebaca1a03b`。
- `uv run python scripts/update_experiment_summary.py`: PASS、135 experiments。
- final `uv run python scripts/validate_experiment.py --experiment exp134_self_gr_multiscale_longtail_gate`: PASS
- post-Kaggle decision logic patch:
  - `uv run python -m py_compile experiments/exp134_self_gr_multiscale_longtail_gate/self_gr_multiscale_longtail_gate.py`: PASS
  - `uv run ruff check experiments/exp134_self_gr_multiscale_longtail_gate/self_gr_multiscale_longtail_gate.py`: PASS
  - `uv run ruff format --check experiments/exp134_self_gr_multiscale_longtail_gate/self_gr_multiscale_longtail_gate.py`: PASS
  - `uv run python scripts/validate_experiment.py --experiment exp134_self_gr_multiscale_longtail_gate`: PASS

## Kaggle train v1

- canonical kernel id: `kentookumura/exp134-self-gr-gate-train`
- URL: `https://www.kaggle.com/code/kentookumura/exp134-self-gr-gate-train`
- push: `Kernel version 1 successfully pushed`
- existence check: `kaggle kernels pull kentookumura/exp134-self-gr-gate-train -p /tmp/kaggle-pull/exp134-self-gr-gate-train-v1 -m`: PASS
- log follow was started, then stopped per user instruction. User will notify when Kaggle run completes.
- user completion notice received; logs and output fetched.
- output: `experiments/exp134_self_gr_multiscale_longtail_gate/kaggle/output/train_v1`
- rows / wells: 3,783,989 / 773
- runtime: 3315.821 sec
- best reference prediction: `pred_exp092_lgb1` RMSE 9.322479896
- baseline `pred_likpf_mean`: RMSE 11.594898
- direct `pred_tvt_dense`: RMSE 23.470396
- best self-GR gate: `pred_dense_longtail_pf_dense_q4_self_q75` RMSE 15.304252、`likpf_mean` から +3.709355 悪化、gate rate 0.058405
- no-self dense gate: `pred_dense_longtail_pf_dense_q4` RMSE 22.105710、`likpf_mean` から +10.510812 悪化、gate rate 0.223399
- common-worst top26: `likpf_mean` RMSE 36.823806、`tvt_dense` 20.539466、no-self dense gate 21.537070、self_q75 gate 32.578503
- worst-well regression for self_q75 gate: max +96.835970 RMSE
- decision: direct self-GR gate rejected。self-GR 条件は dense gate の破壊を一部抑えるが、overall / worst-well が大きく悪化するため inference port / submit / standalone follow-up はしない。
- input decompressed cache SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- gate predictions decompressed SHA: `132288cd7dfa93d62c3ddae0ba182668d3790ebd451b3c285f72fb10ce8604b0`
- copied small generated files to `experiments/exp134_self_gr_multiscale_longtail_gate/artifacts/`; large `gate_predictions.csv.gz` remains only under Kaggle output.
- note: Kaggle v1 output summary JSON treated reference `pred_exp092_lgb1` as best for the automatic recommendation. Local script was patched after output fetch so future runs base `decision` on configured self-GR gate variants only; repo result files record the corrected rejected decision.

## 未実行

- local train / notebook execution は AGENTS ルールに従い未実行。
