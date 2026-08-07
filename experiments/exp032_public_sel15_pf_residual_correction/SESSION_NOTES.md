# exp032_public_sel15_pf_residual_correction セッションノート

## 目的

`exp029` の public sel15 PF/Beam OOF-like artifact を使い、`target_tvt - pf_pred` を直接学習する residual correction が fold 外でも安定するか監査する。

## 現在の状態

- Route: pf_beam
- 状態: 完了
- CV: 14.937393 (`ridge_residual_shrink0p5_clip20p0`, original-fold OOF)
- LB: なし
- 親: `exp029_public_sel15_pf_oof_feature_generation`
- Kaggle kernel: `kentookumura/exp032-sel15-pf-residual-train` version 2
- output path: `/tmp/kaggle-output/exp032_public_sel15_pf_residual_correction/train_v2`
- artifact path: `experiments/exp032_public_sel15_pf_residual_correction/artifacts/`

## コマンドログ

### 実行済み

```bash
uv run python scripts/new_steering.py --experiment exp032_public_sel15_pf_residual_correction
uv run python scripts/new_experiment.py --name exp032_public_sel15_pf_residual_correction --source experiments/exp030_public_sel15_pf_candidate_selector
uv run ruff check experiments/exp032_public_sel15_pf_residual_correction/residual_correction_audit.py
uv run python scripts/validate_experiment.py --experiment exp032_public_sel15_pf_residual_correction
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp032_public_sel15_pf_residual_correction --notebook train --kernel-id kentookumura/exp032-sel15-pf-residual-train --title "exp032 sel15 pf residual train" --run-on-push --strict
kaggle kernels push -p experiments/exp032_public_sel15_pf_residual_correction/kaggle/train
kaggle kernels pull kentookumura/exp032-sel15-pf-residual-train -p /tmp/kaggle-pull/exp032-sel15-pf-residual-train -m
kaggle kernels logs kentookumura/exp032-sel15-pf-residual-train
kaggle kernels output kentookumura/exp032-sel15-pf-residual-train -p /tmp/kaggle-output/exp032_public_sel15_pf_residual_correction/train_v2
cp /tmp/kaggle-output/exp032_public_sel15_pf_residual_correction/train_v2/metrics.json experiments/exp032_public_sel15_pf_residual_correction/metrics.json
cp /tmp/kaggle-output/exp032_public_sel15_pf_residual_correction/train_v2/artifacts/residual_correction_*.csv /tmp/kaggle-output/exp032_public_sel15_pf_residual_correction/train_v2/artifacts/residual_correction_summary.json experiments/exp032_public_sel15_pf_residual_correction/artifacts/
```

## 変更点

- `config.yaml` を residual correction audit 用に更新。
- `settings.py` の `EXPERIMENT_NAME` を exp032 に更新。
- `residual_correction_audit.py` を追加。
- train / inference notebook を exp032 名にリネームし、audit-only 構成に更新。
- `exp026_oof` は exp029 artifact で全欠損のため特徴量から除外。
- Kaggle kernel source に `kentookumura/exp029-sel15-pf-oof-train` を追加し、Kaggle input から feature CSV を読む fallback を追加。

## 結果

- lint: `residual_correction_audit.py` PASS
- validation: `scripts/validate_experiment.py --experiment exp032_public_sel15_pf_residual_correction` PASS
- Kaggle train v1: FAILED。notebook preview cell が local feature path を直接確認して `FileNotFoundError`。
- Kaggle train v2: COMPLETED。約 260 秒で metrics / artifacts を保存。
- rows: 1,782,279
- wells: 773
- public PF selector: 15.172636
- fixed `pf090_hold010`: 15.089532
- best original-fold OOF: `ridge_residual_shrink0p5_clip20p0` 14.937393
- best well-hash holdout: `ridge_residual_shrink0p5_clip20p0` 14.844228
- selected original-fold delta: -0.235243 vs public PF、-0.152138 vs `pf090_hold010`
- selected well-hash delta: -0.328408 vs public PF、-0.245304 vs `pf090_hold010`
- distance buckets: selected は original-fold / well-hash の全 bucket で public PF より改善
- caution: original-fold split 3 は selected が public PF より +0.074417 悪化

## 次のアクション

1. `ridge_residual_shrink0p5_clip20p0` を exp027 public sel15 inference flow に移植する別実験を作る。
2. 推論 output diff、range、start continuity、submit-check を確認する。
3. Public LB anchor 8.781 に対して提出価値があるか判断する。
