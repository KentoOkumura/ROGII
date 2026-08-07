# 設計

## アプローチ

exp238 の selector artifact と feature builder を読み込み、outer fold ごとに同じ
380 base features + 35 nested rank-slot features を再構成する。各 fold の residual target に対し、
LightGBM の代わりに公開 notebook 由来 CatBoost `cb0` を1本学習する。保存済み
exp238 final OOF を control とし、control 自体は再学習しない。

## 仮説

CatBoost の ordered boosting / symmetric-tree の帰納バイアスが exp238 LightGBM と異なり、
415列 feature surface における model-family diversity を生む。ただし exp012 の過去実績は負のため、
raw guard 不通過時は追加チューニングで救済しない。

## 実験範囲

- 対象実験: `exp274_catboost_final_regressor_swap_on_exp238`
- Route: `ml_model`
- 親実験: `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`
- 変更する変数: final estimator family (`LightGBM` -> `CatBoost`) のみ
- 固定する変数: exp238 row / outer fold / residual target / 380 base features / 35 nested rank-slot features / selector score artifacts

### 公開 notebook 設定

- source: `docs/notebooks/rogii-wellbore-geology-prediction/score_ascending_20260611/pixiux__rogii-dual-pipeline-blend/rogii-dual-pipeline-blend.ipynb`
- source SHA256: `9f80687b9582b9b47a464613433afabe74274565252a2e235c152456a0d828e8`
- config: `iterations=8000`, `depth=7`, `l2_leaf_reg=2.0`, `min_data_in_leaf=15`, `border_count=254`, `loss_function=RMSE`, `task_type=GPU`, `od_type=Iter`, `od_wait=300`, `learning_rate=0.02`, `random_seed=7`
- fit: `early_stopping_rounds=250`, `use_best_model=True`
- 公開 notebook の2本目 `cb1` は実行しない。

## 再現性設計

- seed policy: 公開 notebook `cb0` の `random_seed=7` を固定する。
- stochastic 処理の有無: CatBoost GPU 学習は stochastic かつ bitwise deterministic とは見なさない。特徴生成と selector score は保存済み契約を固定する。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規実行はなし。exp238 が保存した selector score / feature input としてのみ使う。
- 並列処理と乱数の関係: CatBoost 内部 GPU 実行に限定し、Python global RNG を使わない。
- CPU/GPU runtime と deterministic flags: Kaggle T4 GPU、single device `0`。公開設定の `task_type=GPU` を維持する。GPU 差のため deterministic anchor には昇格させない。
- train cache / test feature regeneration の SHA 記録方針: selector score gzip は decompressed SHA、fold ごとの CatBoost 入力 float32 matrixは content SHA、feature schema SHA を記録する。
- model manifest / prediction / submission SHA 記録方針: 5 `.cbm` model SHA、OOF decompressed SHA、metrics / guard SHA を記録する。submission は guard 通過後の別ターンで記録する。
- Kaggle package bootstrap 確認方針: prepare 後に bootstrap 内 `config.yaml`、kernel sources、GPU metadata、CatBoost config の一致を検査する。

## リスク

- リークリスク: outer fold の train/valid role とそれに対応する nested score を厳密に照合する。valid target/error を特徴へ追加しない。
- CV/LB 不一致リスク: exp257 は CV/LB 逆転があるため、Public LB ではなく exp238 same-fold OOF を主比較にする。
- ランタイム/メモリリスク: 約378万行 x 415特徴の CatBoost Pool は CPU/GPU memory が大きい。fold を1本ずつ実行し、Pool / matrix / model を毎回解放する。
- 再現性リスク: CatBoost GPU の rerun が bitwise 一致するとは仮定しない。kernel version / model / prediction SHA を実行ごとに保存する。

## 次

1 config x 5 folds = 5 CatBoost models、1 model あたり最大8,000 iterations、理論上最大40,000 trees、
parent/control retraining 0 の契約を提示し、明示承認後にのみ Kaggle GPU train を実行する。

## Reference inference override（2026-07-18）

- authorization: ユーザーの「参考のため推論も行ってください」をreference-only inference承認として記録する。
- adoption boundary: `all_raw_guards_pass=false`と`inference_allowed=false`は変更しない。生成物は採用候補・提出候補に昇格させない。
- feature generation: exp238 hidden-safe inferenceと同じcurrent-test replay、dynamic exp226、multiobs、exp145 learned、GRWR、保存済みouter5 x inner4 selectorを使い、380 + 35 = 415列を再生成する。
- final models: exp274 train version 1の保存済みCatBoost 5 `.cbm`をfold-matched rank-slot featuresで推論する。学習は0。
- comparison: 同じ415列matrixへ保存済みexp238 LightGBM 15 modelsを適用してparent referenceを再生成し、固定`0.75 * parent + 0.25 * CatBoost`も保存する。weight探索はしない。
- primary output: root `submission.csv`はraw CatBoost。parent/blendは`artifacts/`配下のreference CSVとし、competition submitは行わない。
- reproducibility: current-test matrix SHA、model SHA、prediction decompressed SHA、3 CSVのSHA、kernel version、fallback rows、prediction rangeを記録する。
