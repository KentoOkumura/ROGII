# exp066_cv_submit_gate

## 状態

- ルート: ml_model
- 状態: completed
- CV: 9.630105 (source exp063 inference candidate)
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-13
- 親実験: exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit

## 仮説

`exp063` の strict Pixiux public replay は CV、推論完了、submit-check、train/test overlap probe の条件を満たしており、1 回の code submit で Public LB を確認する価値がある。

## 変更点

- 新しい学習・推論は行わず、`exp063` inference v2 を提出候補として gate する。
- `exp064` の no-overlap probe 結果を leakage 補助条件に入れる。
- 判定結果を `artifacts/cv_submit_gate_decision.json`、`artifacts/cv_submit_gate_decision.csv`、`artifacts/cv_submit_gate_report.md` に保存する。

## 検証方針

- Fold: `exp063` の GroupKFold 5 folds を参照
- Group: public well id
- Stratification: なし
- Leakage Check: `exp064` hidden code submission の assertion が発火しなかったこと、`exp063` inference が static visible override / hidden-specific branch を含まないことを確認

## 実行入口

- 学習 notebook: `exp066_cv_submit_gate_train.ipynb`
- 推論 notebook: `exp066_cv_submit_gate_inference.ipynb`
- Kaggle 準備: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp066_cv_submit_gate --notebook train --strict`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| source CV | 9.630105 |
| gate decision | approved_for_code_submit |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- `exp063` inference v2 は complete、submit-check PASS、fallback 0、14,151 rows。
- ローカル submission SHA は `exp063` metrics と一致した。
- `exp064` の code submission probe は hidden scoring test で train/test well_id overlap assertion が発火しなかった。

### 悪かった点

- Public LB は未確認。gate 通過は LB anchor 更新ではない。

### リスク / 注意

- `exp063` の CV surface と Public LB は一致しない可能性がある。
- 提出は `exp063` inference kernel version 2 を対象に行う。`exp066` を提出 kernel として使わない。

## 次

- 提出回数を使う判断をしたら、`kentookumura/exp063-ravaghi-pixiux-strict-replay-infer` v2 の `submission.csv` を code submit する。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
