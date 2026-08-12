# 要件

## 依頼

`exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218` の最終 TVT 回帰器だけを
LightGBM から CatBoost に差し替える。CatBoost のハイパーパラメータは保存済み公開
notebook `pixiux/rogii-dual-pipeline-blend` の先頭 CatBoost config (`cb0`) をそのまま使う。

2026-07-18 追記: train-side raw guard は不通過だが、ユーザーの明示依頼により参考値として
current-test inference を実行する。これは採用・提出判断ではなく、raw CatBoost と固定0.25
blendの予測分布を確認するためのreference-only overrideとする。

## 仮説

exp238 の nested rank-slot feature surface では、公開 CatBoost `cb0` が現行 LightGBM と
補完的な誤差を持ち、raw または事前固定0.25 blend で same-fold OOF を改善する。

## 制約

- Route: `ml_model`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- exp238 の selector score、rank-slot 特徴、380列 base feature、outer fold、residual target は固定する。
- selector 20 boosters と exp238 final LightGBM 15 boosters は再学習しない。
- 新規学習は CatBoost `cb0` 1 config x outer 5 folds = 5 models のみ。
- 予測値は保存済み exp238 `lgb_mean` OOF と比較する。
- 固定 `CatBoost 0.25 + exp238 LightGBM 0.75` blend は多様性診断として1回だけ評価する。
- 初回 notebook 実行は Kaggle GPU で行う。この実装ターンでは push しない。

## 受け入れ基準

- 公開 notebook `cb0` の model params と fit-time early stopping が `config.yaml`、notebook output、parameter audit で一致する。
- 1 variant / 1 config / 5 folds / 5 CatBoost models、parent/control retraining 0 が機械的に検査される。
- exp238 の row order、fold role、feature schema、selector score artifact を fail-closed で検査する。
- raw CatBoost と固定0.25 blendについて overall、fold、distance bucket、hidden-like 2面、by-well を出力する。
- overall、1000+、hidden-like 2面、worst-well、fold 数の guard は採用条件として維持する。guard不通過時のinferenceは、ユーザー明示承認・reference-only・competition submitなしの場合だけ許可する。
- reference inferenceでは保存済みCatBoost 5 modelsだけを使い、学習・再学習を行わない。
- current testからexp218 base 380列とfold-matched exp238 selector rank-slot 35列をhidden-safeに再生成する。
- primary `submission.csv` はraw CatBoostとし、保存済みparent LightGBMと固定0.25 blendは比較用生成物に分離する。
- sample submissionとID/row order、NaN/Inf、fallback 0、feature schema 415列、model SHAをfail-closedで検査する。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。

## 次

reference-only current-test portを同じexp274内へ実装し、Kaggle inference outputを取得して
submit-checkする。competition submitは別途明示依頼がない限り行わない。
