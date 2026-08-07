# 設計

## アプローチ

旧 exp063 の exp056 artifact ベース audit は公開 notebook 再現として不十分だったため無効化する。
新しい実装では、Pixiux public notebook 内に含まれる public Ravaghi-style base feature builder と
likelihood-PF feature builder を `public_notebook_replay_audit.py` に同梱し、
competition raw train files から train-side features を再生成する。

比較は次の 2 variant に限定する。

- `ravaghi_public_lgbm_replay`: public base features
- `pixiux_likpf_public_replay`: public base features + likelihood-PF delta features

学習は public LightGBM 3 configs x GroupKFold で実行する。
各 fold / 各 LightGBM config の `feature_importances_` を保存し、
学習後に `variant, feature` 単位で平均した CSV と matplotlib PNG を出力する。

## 実験範囲

- 対象実験: `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`
- Route: `ml_model`
- 親: public notebooks
- 実装元: `public_notebook_replay_audit.py`
- 固定する変数: train wells、GroupKFold、target、public LightGBM configs、後処理なし。
- 除外する要素: CatBoost、Ridge stack、final blend、projection、pretrained booster、static visible override、test-side submission logic。

## Inference port

追加の inference notebook は、train audit で最良だった
`pixiux_likpf_public_replay` `lgb_mean` だけを対象にする。
train notebook は public LightGBM 3 configs x GroupKFold の 15 fold booster を保存する。
inference notebook は saved booster を読み、raw test files から public base features と likelihood-PF features だけを生成する。
各 saved booster の test residual prediction を平均し、`last_known_tvt + pred_delta` を `submission.csv` に保存する。

後続実験の train で再利用できるよう、PF/Beam/likelihood-PF tracker features を compact csv.gz として保存する。
保存列は `id`, `well`, `target`/`last_known_tvt` と、`pf_`, `beam_`, `sc*`, `hyb_`, `sig_`, `likpf_`, `tdpf*`, `tdbc*` 系の tracker columns に限定する。

この port は hidden-specific branch ではなく、公開 replay 条件の saved-model direct inference である。
guarded overlap override、static visible override、pretrained boosters、CatBoost、Ridge stack、
final public notebook blend、projection postprocess は入れない。

## リスク

- ランタイムリスク: Pixiux likelihood-PF は 128 seeds x 500 particles で重い。Kaggle GPU を有効にして実行し、ログで進捗を確認する。
- 再現境界リスク: full submission notebook の final blend までは再現しない。目的は LightGBM 入力特徴量の train-side 比較に限定する。
- CV/LB 不一致リスク: train-side audit なので LB anchor は更新しない。strict replay OOF が出るまで exp063 の旧結果は採用判断に使わない。
- Inference runtime リスク: test feature 生成は重いが、train feature 再生成と inference-time LightGBM training はしない。saved booster がない既存 train v3 output では動かないため、train を一度 rerun して artifact を作る。
