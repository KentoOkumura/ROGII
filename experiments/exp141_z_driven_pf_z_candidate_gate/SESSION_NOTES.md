# exp141_z_driven_pf_z_candidate_gate セッションノート

## 目的

`likpf_mean` を default に固定し、Z-driven と見なせる低頻度区間だけ `pf_z` を候補として選ぶ target-free gate を検証する。

## 現在の状態

- Route: `pf_beam`
- 状態: `completed_train_side_rejected_no_submit`
- CV: 11.594897672
- LB: まだなし
- blocked: none

## コマンドログ

```bash
uv run python scripts/new_steering.py --experiment exp140_z_driven_pf_z_candidate_gate
uv run python scripts/new_experiment.py --name exp140_z_driven_pf_z_candidate_gate
mv experiments/exp140_z_driven_pf_z_candidate_gate experiments/exp141_z_driven_pf_z_candidate_gate
mv docs/legacy/steering/20260627-exp140-z-driven-pf-z-candidate-gate docs/legacy/steering/20260627-exp141-z-driven-pf-z-candidate-gate
uv run python -m py_compile experiments/exp141_z_driven_pf_z_candidate_gate/z_driven_pf_z_candidate_gate.py experiments/exp141_z_driven_pf_z_candidate_gate/settings.py
python3 -m json.tool experiments/exp141_z_driven_pf_z_candidate_gate/exp141_z_driven_pf_z_candidate_gate_train.ipynb
python3 -m json.tool experiments/exp141_z_driven_pf_z_candidate_gate/exp141_z_driven_pf_z_candidate_gate_inference.ipynb
uv run python scripts/validate_experiment.py --experiment exp141_z_driven_pf_z_candidate_gate
uv run ruff check experiments/exp141_z_driven_pf_z_candidate_gate/z_driven_pf_z_candidate_gate.py experiments/exp141_z_driven_pf_z_candidate_gate/settings.py
uv run ruff format --check experiments/exp141_z_driven_pf_z_candidate_gate/z_driven_pf_z_candidate_gate.py experiments/exp141_z_driven_pf_z_candidate_gate/settings.py
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp141_z_driven_pf_z_candidate_gate --notebook train --kernel-id kentookumura/exp141-z-pfz-gate-train --title 'exp141 z pfz gate train' --run-on-push --strict
python3 -m json.tool experiments/exp141_z_driven_pf_z_candidate_gate/kaggle/train/exp141_z_driven_pf_z_candidate_gate_train.ipynb
uv run python -m py_compile experiments/exp141_z_driven_pf_z_candidate_gate/kaggle/train/z_driven_pf_z_candidate_gate.py experiments/exp141_z_driven_pf_z_candidate_gate/kaggle/train/settings.py
kaggle kernels push -p experiments/exp141_z_driven_pf_z_candidate_gate/kaggle/train
kaggle kernels status kentookumura/exp141-z-pfz-gate-train
kaggle kernels output kentookumura/exp141-z-pfz-gate-train -p experiments/exp141_z_driven_pf_z_candidate_gate/kaggle/output/train_v1
kaggle kernels logs kentookumura/exp141-z-pfz-gate-train > experiments/exp141_z_driven_pf_z_candidate_gate/kaggle/output/train_v1/exp141-z-pfz-gate-train.log
```

## 変更点

- 当初 `exp140_z_driven_pf_z_candidate_gate` として作成したが、既存 `exp140_z_slope_posthoc_correction_on_pfbeam_candidates` と番号衝突していたため、`exp141_z_driven_pf_z_candidate_gate` に改番した。
- 旧番号で誤って Kaggle に push した kernel `kentookumura/exp140-z-pfz-gate-train` version 1 は、誤番号 run として扱い、この実験の正式結果には使わない。
- `docs/legacy/steering/20260627-exp141-z-driven-pf-z-candidate-gate/` に改番。
- `experiments/exp141_z_driven_pf_z_candidate_gate/` に改番。
- `config.yaml` を `pf_beam` route、exp072 cache parent、low-frequency `pf_z` gate grid に更新。
- `z_driven_pf_z_candidate_gate.py` を追加。
  - exp072 cache から `likpf_mean`、`pf_z`、`pf_ancc`、`beam_mean`、`dzdmd`、`md_since` などを必要列だけ読み込む。
  - `pf_z` slope と `-dZ/dMD` の alignment margin、候補差分、PF/Beam disagreement、roughness guard を計算する。
  - row / segment / well gate と switch-rate cap を評価する。
  - metrics、gate variants、by-well、bucket、representative wells、raw-test parity、summary を保存する。
- train notebook を設定確認、入力確認、gate plan、audit 実行、生成物確認の構成に更新。
- inference notebook は diagnostic-only marker を保存する構成に更新。

## 再現性メモ

- seed policy: `no_new_rng_posthoc_saved_cache_audit`
- stochastic components: 上流 exp072 cache のみ
- CPU/GPU runtime: CPU-only、GPU 不使用
- Kaggle kernel id / version: `kentookumura/exp141-z-pfz-gate-train` / version 1 COMPLETE
- input / feature schema SHA: Kaggle train の `summary.json` に保存予定
- feature content SHA: gzip decompressed content SHA を保存予定
- model manifest / model SHA: 新規モデルなし
- prediction SHA: submission prediction なし
- submission SHA: submission なし
- rerun check: 未実行

## 検証

- `py_compile` for local `z_driven_pf_z_candidate_gate.py` and `settings.py`: PASS
- local train notebook JSON: PASS
- local inference notebook JSON: PASS
- `validate_experiment.py --experiment exp141_z_driven_pf_z_candidate_gate`: PASS
- `ruff check` for exp141 script and settings: PASS
- `ruff format --check` for exp141 script and settings: PASS
- synthetic frame smoke test for `build_surface()` + `run_gate_audit()`: PASS
- `prepare_kaggle_notebooks.py --notebook train --strict`: PASS
- packaged train notebook JSON: PASS
- packaged `z_driven_pf_z_candidate_gate.py` and `settings.py` py_compile: PASS
- generated train package: `experiments/exp141_z_driven_pf_z_candidate_gate/kaggle/train`
- generated metadata:
  - kernel id: `kentookumura/exp141-z-pfz-gate-train`
  - title: `exp141 z pfz gate train`
  - GPU: false
  - internet: false
  - run_on_push: true
  - competition source: `rogii-wellbore-geology-prediction`
  - kernel source: `kentookumura/exp072-exp063-full-replay-feature-cache-train`
- bootstrap SHA:
  - `config.yaml`: `8cc0925c93816525bbecd31ad0f7e563b363741c488676587f9b999eedfd074b`
  - `z_driven_pf_z_candidate_gate.py`: `0d5def9c0e5877cc9d32c21f21dace4a8b24264094fead329c801c2156fbcb35`
  - `settings.py`: `35b9d65121138d9c951338b484bf7af67cf4f5e0b2aef0341c5f8039a89b2645`
- Kaggle push:
  - command: `kaggle kernels push -p experiments/exp141_z_driven_pf_z_candidate_gate/kaggle/train`
  - result: Kernel version 1 successfully pushed
  - URL: https://www.kaggle.com/code/kentookumura/exp141-z-pfz-gate-train
  - monitoring: ユーザー指示により未実施
- Kaggle output:
  - status: `KernelWorkerStatus.COMPLETE`
  - output: `experiments/exp141_z_driven_pf_z_candidate_gate/kaggle/output/train_v1`
  - logs: `experiments/exp141_z_driven_pf_z_candidate_gate/kaggle/output/train_v1/exp141-z-pfz-gate-train.log`

## 結果

- rows / wells: 3,783,989 / 773
- baseline `likpf_mean`: RMSE 11.594897672、MAE 7.067632675、within10 0.772807479
- `single_pf_z`: RMSE 17.788171172、delta +6.193273499
- `single_pf_ancc`: RMSE 14.493050690、delta +2.898153017
- `single_beam_mean`: RMSE 15.774327032、delta +4.179429360
- oracle `likpf_mean + pf_z`: RMSE 9.115200716、delta -2.479696957
- oracle core PF/Beam: RMSE 6.953036836、delta -4.641860836
- best configured gate: `seg_zq75_alignq60_diffq70_sr010_min32_clip20_a050`
  - RMSE 11.633719432
  - delta vs `likpf_mean`: +0.038821760
  - gate rate: 0.084827
  - max well regression: +4.8418
- raw-test parity checklist:
  - required columns: pass
  - gate target-free: pass
  - new model training: pass
  - PF particle regeneration: pass
  - inference port: not applicable

## 解釈

- `pf_z` は direct replacement では弱く、全体では `likpf_mean` から大きく悪化した。
- oracle は大きいので候補集合としての headroom はあるが、今回の target-free low-frequency gate は選択精度が足りなかった。
- `ba48188d` / `fef8af96` のように `pf_z` が強い代表 well はある一方、`91b301ce` では `pf_z` が悪化し、`pf_ancc` / oracle core のほうが良い。
- row-wise gate は step >= 10 / 25 の不連続を増やし、well-level gate は最大 well regression が大きい。
- 結論: train-side rejected。raw-test inference port、submission、submit-check は行わない。

## 次のアクション

1. `result.md`、`metrics.json`、`experiment_summary.md`、`KAGGLE_DIRECTION.md` は完了結果で更新済み。
2. `z_driven_pf_z_candidate_gate` backlog は完了 / 不採用として閉じた。
3. `pf_z` を使う場合は hard switch ではなく、segment-level verifier / confidence feature / `z_slope_posthoc_correction_on_pfbeam_candidates` の小補正方向に限定する。
