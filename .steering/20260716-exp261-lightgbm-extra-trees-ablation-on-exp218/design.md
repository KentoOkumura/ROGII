# 設計

## アプローチ

exp218の保存済みfeature/cache生成ロジックをbootstrap dependencyとして読み、notebook上で
exp072 base cache、known-prefix anchor、U-projection、exp145 learned-likelihood、GRWR特徴を
順に再構築する。親の `exp063_lgb_config_family()` とGPU mode overrideを読み、選択した
config indexだけに `extra_trees=True` を加える。対応する親configとの差分が
`extra_trees` だけであることをassertしてから、同一well GroupKFoldで新規variantのみ学習する。

実行プランは以下の二つをconfigに定義し、`selected_plan` と `run_approved` の双方が
設定されない限り学習前に停止する。

- `lgb1_probe`: exp218単体CV最良のindex 1だけ、5 boosters。
- `full_family`: index 0/1/2、15 boosters。

親exp218 OOF predictionは再学習せずKaggle kernel sourceから読み、ID/well/target/base予測を
検証する。新規OOFとの相関、対応するhistorical scalar、parent `lgb_mean`との固定blend、
distance bucket、exp115 hidden-like 2面、fold、by-well、worst-wellを保存する。

## 実験範囲

- 対象実験: `exp261_lightgbm_extra_trees_ablation_on_exp218`
- Route: `ml_model`
- 親実験: `exp218_gr_wavelet_rotation_confidence_features_on_exp148`
- 変更する変数: 選択したLightGBM configの `extra_trees=False/未指定 -> True`。
- 固定する変数: feature schema、row surface、target/base prediction、fold、seed、configのその他全parameter、GPU mode、early stopping、metric。
- 初回範囲外: control再学習、selector LightGBM、feature変更、seed grid、parameter grid、inference、submit。

## 再現性設計

- seed policy: exp218のconfig別seed (`123/0/29`) とGroupKFoldを固定する。
- stochastic処理の有無: 新規feature RNGなし。LightGBMのrandom threshold selectionとGPU学習はstochastic componentとして記録する。
- PF/Beam / likelihood-PF / seed baggingの有無: 新規生成なし。exp072/exp145の保存済みcacheだけを使う。
- 並列処理と乱数の関係: `deterministic=true`、`force_col_wise=true`、`gpu_use_dp=true`、`n_jobs=num_threads=8`を親から継承する。GPU bitwise一致はrerunなしに主張しない。
- CPU/GPU runtimeとdeterministic flags: Kaggle GPUを正とし、ローカルnotebookは実行しない。
- train cache SHA: 入力path、row/well/feature count、feature schema SHA、parent OOF decompressed SHAを保存する。
- model manifest / prediction SHA: 各booster file SHA、OOF content SHA、manifest SHA、metrics SHAを保存する。submission SHAは初回範囲外。
- Kaggle package bootstrap確認方針: prepare後にbootstrap内configのselected plan、approval、kernel sources、GPU/internet、dependency fileを確認する。

## リスク

- リークリスク: 親exp218と同じwell GroupKFoldを維持し、hidden-tail targetをfeature生成に使わない。親OOFは評価・blend readoutにのみ使う。
- CV/LB不一致リスク: exp218自体にもbucket / worst-well差があり、extra treesのCV改善がLBへ転移する保証はない。hidden-likeとworst-wellをguardする。
- ランタイム/メモリリスク: full familyは3M行×380特徴×15 boosters。foldごとに行列/modelを解放し、未承認runはfeature構築前に停止する。
- 再現性リスク: `extra_trees=True` がrandom thresholdを使うため、固定seedとdeterministic flagsがあってもGPU bitwise再現性を仮定しない。
