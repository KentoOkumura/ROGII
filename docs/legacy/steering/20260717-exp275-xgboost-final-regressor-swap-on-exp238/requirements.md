# 要件

## 依頼

`exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218` の最終 TVT 回帰器だけを LightGBM から XGBoost に差し替える。XGBoost のハイパーパラメータは、ユーザーが選択した公開 notebook `cdeotte/xgb-starter-cv-15` version 3 の full-run `XGB_PARAMS` を使う。

## 制約

- Route: `ml_model`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 特徴、residual target、outer fold、sample weight、nested selector score、rank-slot 変換は `exp238` と同一にする。
- 学習するのは XGBoost 1 config × 5 folds = 5 boosters だけとする。
- 保存済み `exp238` LightGBM OOF を比較基準に使い、親/control LightGBM と selector は再学習しない。
- XGBoost パラメータ探索、early stopping 追加、blend weight 探索、inference、submission は実装範囲外とする。
- notebook 初回実行は Kaggle GPU を正とし、このターンでは push / train 実行しない。

## 受け入れ基準

- Jupytext percent 形式の compact self-contained train / disabled inference を作成し、読める `.ipynb` へ変換する。
- 公開 notebook の `n_estimators=450`, `learning_rate=0.035`, `max_depth=5`, `min_child_weight=20`, `subsample=0.85`, `colsample_bytree=0.85`, `reg_lambda=4.0`, `reg_alpha=0.05`, `objective=reg:squarederror`, `eval_metric=rmse`, `tree_method=hist`, `max_bin=256`, `random_state=42`, `device=cuda` を設定と parameter audit に固定する。
- `exp238` の 380 base + 35 selector = 415 特徴、3,783,989 rows、773 wells、5 outer folds を fail-closed で検証する。
- raw XGBoost と保存済み `exp238/lgb_mean` の overall、fold、distance bucket、1000+、hidden-like 2面、by-well、worst-well、予測相関を比較する。
- 事前固定 XGBoost weight 0.25 の blend は多様性 readout として1回だけ計算し、選択や weight tuning に使わない。
- 実行前状態で variant / config / fold / booster 数と親/control再学習なしを `SESSION_NOTES.md` に記録する。
- Jupytext `--test`、`py_compile`、`ruff --select F821`、`validate-exp` を通す。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。

## 2026-07-18 参考推論・スコアリング override

- ユーザーの明示依頼により、train-side raw guard FAILを維持したまま参考用current-test推論とKaggleスコアリングを行う。
- 採用判断は変更しない。primary submissionは保存済みXGBoost 5モデルの単純平均だけとし、parent LightGBMと固定0.25 blendは比較生成物に限定する。
- 推論中の学習、selector再学習、parameter / blend weight探索、public-test row artifactの利用は禁止する。
- 保存済みtrain summary / model manifest / feature schema / 5 model SHAをfail-closedで確認する。
- current-test 380 base + outer-fold matched 35 selector rank-slot特徴を再生成し、fallback 0、sample submission完全一致を必須とする。
- Kaggle outputの`submission.csv`をsubmit-checkでPASSした後に限り、raw XGBoostを1件だけsubmitしてPublic LBを参考値として記録する。
