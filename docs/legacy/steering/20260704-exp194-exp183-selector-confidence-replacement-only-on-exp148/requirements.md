# 要件

## 依頼

`exp183_selector_confidence_replacement_only_on_exp148` backlog を実装する。exp188 の add-only negative を受け、exp183 selector confidence features を exp148 の `learned_likelihood_confidence` block と置換する train-side ML 実験として切る。

## 制約

- Route: `ml_model`
- 実験 ID: `exp194_exp183_selector_confidence_replacement_only_on_exp148`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- selector 親: `exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector`
- active variant は 1 つだけにし、control / parent 再学習はしない。
- active feature groups は `projection_correction`、`u_disagreement`、`exp183_selector_confidence` とする。
- `learned_likelihood_confidence` は active variant から除外し、coverage / inventory 以外では使わない。
- exp183 selected TVT を直接の prediction replacement、blend、postprocess、hard gate、submission 候補として使わない。
- 再現性: `docs/06_reproducibility.md` に従い、upstream cache、GPU LightGBM、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

- `docs/legacy/steering/`、`experiments/exp194.../config.yaml`、train / inference notebook source、`SESSION_NOTES.md`、`result.md`、`metrics.json` が作成されている。
- train notebook が active variant、feature groups、planned booster 数を表示する。
- active variant から `learned_likelihood_confidence` が外れている。
- 予定 train cost が 1 variant x 3 LightGBM configs x 5 folds = 15 boosters と記録されている。
- Jupytext 変換、`py_compile`、`ruff --select F821`、`make validate-exp` が通る。
- deterministic anchor として扱わない。実行後に採用候補にする場合は、feature content SHA、model manifest SHA、prediction SHA、Kaggle kernel version を記録する。
