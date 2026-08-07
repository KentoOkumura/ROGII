# 要件

## 依頼

`exp183_selector_confidence_addonly_on_exp148` を実験化し、実験開始時点の番号として `exp188_exp183_selector_confidence_addonly_on_exp148` を付与する。

## 制約

- Route: `ml_model`
- 親実験は `exp148_learned_likelihood_fulltrain_addonly_on_exp092`。
- exp148 の `projection_correction`、`u_disagreement`、`learned_likelihood_confidence` feature surface は固定する。
- 追加するのは exp183 の OOF best-Viterbi selector selected-path confidence feature のみ。
- exp183 selected TVT は direct replacement、blend、postprocess、hard gate、submit candidate として使わない。
- Kaggle train は GPU runtime とし、train notebook は lgb0/lgb1/lgb2 に分割しない。
- control / parent retraining はしない。exp148 CV 8.501281182 / Public LB 7.960 を historical baseline として参照する。
- 再現性: `docs/06_reproducibility.md` に従い、GPU 学習、upstream OOF artifact、Kaggle bootstrap、SHA 記録の扱いを記録する。

## 受け入れ基準

- `.steering/20260704-exp188-exp183-selector-confidence-addonly-on-exp148/` が作成され、狙い・設計・タスクが記録されている。
- `experiments/exp188_exp183_selector_confidence_addonly_on_exp148/` に config、train/inference notebook、実装 module、README、SESSION_NOTES、result、metrics がある。
- train は 1 active variant、3 LightGBM configs、5 folds、合計 15 boosters を予定し、control 再学習なしである。
- `exp183` OOF artifact から `true_tvt`、`abs_error`、`oracle_candidate`、`oracle_label` を downstream feature に入れない。
- Jupytext 変換、構文チェック、`ruff --select F821`、`make validate-exp` が通る。
- Kaggle push 前に GPU cost guard と bootstrap/metadata を確認し、`SESSION_NOTES.md` に記録する。
