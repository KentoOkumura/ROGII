# exp066_cv_submit_gate セッションノート

## 目的

`exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit` の strict Pixiux replay inference v2 を code submit してよいか、CV・推論完了・submit-check・train/test overlap probe の根拠で判定する。

## 現在の状態

- Route: ml_model
- 状態: completed
- CV: 9.630105 (`exp063` inference candidate)
- LB: 未提出
- Gate: approved_for_code_submit

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 実行済み

```bash
uv run python scripts/new_steering.py --experiment exp066_cv_submit_gate
uv run python scripts/new_experiment.py --name exp066_cv_submit_gate
uv run python -m py_compile experiments/exp066_cv_submit_gate/cv_submit_gate.py experiments/exp066_cv_submit_gate/settings.py
uv run python -m json.tool experiments/exp066_cv_submit_gate/exp066_cv_submit_gate_train.ipynb
uv run python -m json.tool experiments/exp066_cv_submit_gate/exp066_cv_submit_gate_inference.ipynb
uv run python experiments/exp066_cv_submit_gate/cv_submit_gate.py
```

## 変更点

- `cv_submit_gate.py` を追加。
  - `exp063` metrics と source submission を読み、CV、inference status、submit-check、fallback rows、row count、SHA256、予測範囲を判定する。
  - `exp064` metrics を読み、hidden scoring test で train/test well_id overlap assertion が発火しなかったことを判定条件に入れる。
  - `artifacts/cv_submit_gate_decision.json`、`artifacts/cv_submit_gate_decision.csv`、`artifacts/cv_submit_gate_report.md` を保存する。
- train / inference notebook を gate audit 用の読める構成に更新。
- `exp066` 自体は submission.csv を生成せず、提出対象は `exp063` inference kernel v2 として記録する。

## 結果

- Decision: `approved_for_code_submit`
- Required rules: 11/11 PASS
- Source CV: 9.630105123038494
- Pixiux vs Ravaghi mean delta: -0.930432 RMSE 相当の改善
- Source submission: `/tmp/kaggle-output/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/infer_v2/submission.csv`
- Source SHA256: `36486e2e5a049ae02b51daa2a06e317bc6c7b841d5fe25841427b792a24f2499`
- Submit target: `kentookumura/exp063-ravaghi-pixiux-strict-replay-infer` version 2, output file `submission.csv`

提出コマンド候補:

```bash
kaggle competitions submit rogii-wellbore-geology-prediction -k kentookumura/exp063-ravaghi-pixiux-strict-replay-infer -v 2 -f submission.csv -m "exp063 strict Pixiux replay lgb_mean CV 9.630105; exp066 gate approved"
```

## 次のアクション

1. 提出回数を使う判断をしたら、上記の `exp063` inference v2 を code submit する。
2. 提出後は `submissions/SUBMISSIONS.md`、`exp063` / `exp066` の result、`experiment_summary.md`、`KAGGLE_DIRECTION.md` に Public LB と解釈を記録する。
