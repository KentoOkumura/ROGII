# 要件

## 依頼

`state_known_tvt_self_gr_hmm_emission` backlog を実装する。exp209 exact HMM の typewell GR emission に対し、self-GR を HMM candidate state が known-prefix TVT 範囲内に入る場合だけ弱く足す train-side diagnostic を作る。

## 制約

- Route: `ensemble`
- self-GR を candidate TVT、replacement、hard switch、postprocess として使わない。
- HMM candidate state `grid[j]` ごとに known-prefix TVT 範囲内かを判定し、範囲外 state は self-GR neutral とする。
- train OOF では評価行より後ろの `TVT_input` を見ない。self-GR curve は finite `TVT_input` と finite `GR` の known prefix のみで作る。
- 初回 active variant は `alpha=0.07`、`clip=1.0`、`boost_only` の 1 本に限定する。
- LightGBM 学習、fold 学習、control 再学習、GPU 使用、inference、submit はしない。
- 再現性: `docs/06_reproducibility.md` に従い、HMM no RNG、outer parallel の floating tolerance、gzip decompressed content SHA の扱いを記録する。

## 受け入れ基準

- `experiments/exp225_state_known_tvt_self_gr_hmm_emission/` に config、train/inference notebook、HMM helper、比較 helper、README、SESSION_NOTES、result、metrics が揃っている。
- `config.yaml` に `experiment.route: ensemble`、親実験、active variant 数、0 booster 方針、state-known self-GR 設定が明記されている。
- HMM emission 実装で、known-prefix TVT 範囲外の candidate state に self-GR boost が足されない。
- train notebook 上で active variants、LightGBM config count、fold count、booster count、parent/control retraining の有無が表示される。
- `py_compile`、`ruff --select F821`、`validate_experiment.py`、Jupytext 変換と `--test` が通る。
- Kaggle train を push する場合は、CPU-only / 1 variant / 0 configs / 0 folds / 0 boosters / control retraining なしを `SESSION_NOTES.md` に記録してから行う。
