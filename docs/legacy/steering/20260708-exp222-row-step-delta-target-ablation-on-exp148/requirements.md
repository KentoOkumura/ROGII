# 要件

## 依頼

`row_step_delta_target_ablation_on_exp148` を実験化し、exp148 の feature surface は固定したまま、教師だけを row-to-row step delta に変える。ユーザー指定により、まずは CPU 実行の `lgb0` のみで反証する。

## 制約

- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 学習対象: 1 enabled variant x 1 CPU mode x `lgb0` x 5 folds = 5 boosters
- 親 exp148 control の再学習はしない。比較は保存済み exp148 / exp193 / exp198 metrics を使う。
- feature surface は exp148 と同一にし、新しい feature は追加しない。
- 既存の anchor residual target `TVT_i - last_known_tvt` は直接学習しない。
- 最初の unknown row は `TVT_i - last_known_tvt`、以降は `TVT_i - TVT_{i-1}` を `target_step_delta` とする。
- OOF 評価では predicted step delta を well ごとに `last_known_tvt + cumsum(pred_step_delta)` で TVT に戻し、復元後 TVT RMSE を primary metric とする。
- 自分の直前予測 TVT / delta、OOF prediction、valid/test true TVT、oracle best、true-error rank、評価 label は feature に入れない。
- 再現性: `docs/06_reproducibility.md` に従い、CPU deterministic flags、seed、input SHA、model/prediction SHA を記録する。

## 受け入れ基準

- `docs/legacy/steering/`、`config.yaml`、train notebook source、helper、`SESSION_NOTES.md`、`result.md` が exp222 として作成されている。
- `config.yaml` に `experiment.route: ml_model`、親、target 差分、CPU runtime、`lgb_config_indices: [0]` が明記されている。
- train notebook は Kaggle runtime を正とし、入力 cache preview、target contract、lgb0 training、metrics / bucket / by-well / cumulative drift / feature importance の表示を含む。
- helper は `target_step_delta` を作り、OOF prediction を well-wise cumulative sum で `pred_tvt` に戻して評価する。
- `py_compile`、`ruff --select F821`、Jupytext 変換/テスト、`validate-exp` が通る。
- Kaggle push 前の計画として、variant 数、mode 数、fold 数、booster 数、control 再学習なしを `SESSION_NOTES.md` に記録する。
