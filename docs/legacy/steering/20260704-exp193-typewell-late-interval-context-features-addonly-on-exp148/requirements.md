# 要件

## 依頼

`typewell_late_interval_context_features_addonly_on_exp148` backlog を `exp193_typewell_late_interval_context_features_addonly_on_exp148` として実装する。

## 制約

- Route: `ml_model`
- 親実験は `exp148_learned_likelihood_fulltrain_addonly_on_exp092`。
- exp148 control / parent は再学習しない。保存済み exp148 CV 8.50128118189582 / Public LB 7.960 を historical baseline とする。
- 追加 feature は raw typewell TVT range と observed finite `TVT_input` prefix の最後だけから作る。
- `candidate_pct_*`、candidate 別 late-range violation / distance / inside flag、exp176 selected TVT、exp148 OOF error、oracle best、true-error rank は使わない。
- PF/Beam prediction、ML prediction、clip、blend、postprocess、hard selector は入れない。
- 再現性: `docs/06_reproducibility.md` に従い、feature source、LightGBM seed / deterministic flags、SHA 記録方針を設計に明記する。

## 受け入れ基準

- `docs/legacy/steering/20260704-exp193-typewell-late-interval-context-features-addonly-on-exp148/` に要件、設計、タスクが記録されている。
- `experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/` に config、settings、train/inference notebook、実装 module、README、SESSION_NOTES、result、metrics がある。
- `config.yaml` の `experiment.route` は `ml_model`。
- active train は 1 variant、1 mode、3 LightGBM configs、5 folds、15 boostersで、control 再学習を含まない。
- 追加 feature group は `typewell_late_interval_context` で、exp148 の既存 projection / learned likelihood features は維持する。
- Jupytext 変換、構文チェック、F821/F401、`validate_experiment.py` が通る。
