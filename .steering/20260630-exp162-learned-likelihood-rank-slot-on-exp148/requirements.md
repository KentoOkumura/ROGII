# 要件

## 依頼

`learned_likelihood_rank_slot_on_exp148` を実装する。実行は CPU とし、exp148 control は再学習しない。

## 制約

- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 保存済み exp148 CV / Public LB を baseline として使う。
- active variant は `learned_likelihood_rank_slot_addonly` の 1 個に限定する。
- active mode は `cpu_deterministic_threads8` のみ。
- Candidate TVT path を hard selector、soft average、blend、postprocess replacement として使わない。
- rank 作成に valid/test true TVT、oracle best、true-error rank を使わない。
- 再現性: `docs/06_reproducibility.md` に従い、upstream cache SHA、feature schema、model manifest、prediction SHA を記録できる構成にする。

## 受け入れ基準

- `experiments/exp162_learned_likelihood_rank_slot_on_exp148/` に config、settings、train/inference notebook、実装 module、README、SESSION_NOTES、result、metrics がある。
- `config.yaml` の `experiment.route` が `ml_model`、`runtime.kaggle.enable_gpu` が `false`、active mode が `cpu_deterministic_threads8` である。
- train notebook が active variant 数 1、LightGBM config 数 3、fold 数 5、合計 15 boosters を表示する。
- exp148 の learned likelihood confidence features を残し、追加 group として learned likelihood rank-slot features を add-only する。
- `make validate-exp EXP=exp162_learned_likelihood_rank_slot_on_exp148` が通る。
