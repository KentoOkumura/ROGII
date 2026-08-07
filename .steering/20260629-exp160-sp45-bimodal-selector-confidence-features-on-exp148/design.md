# 設計

## アプローチ

exp148 の full-train add-only LightGBM flow をコピーし、既存の exp148 feature surface に `sp45_bimodal_selector_confidence` group を追加する。公開 notebook の selector 出力は予測として採用せず、target-free な candidate quality、score margin、candidate spread、prefix trust、near/longtail interaction、U-shape diagnostics へ変換して LightGBM に読ませる。

## 実験範囲

- 対象実験: `exp160_sp45_bimodal_selector_confidence_features_on_exp148`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 変更する変数: SP45/Bimodal confidence feature group の追加
- 固定する変数: exp072/exp092 base features、U-projection settings、exp145 learned likelihood feature config、LightGBM config family、GroupKFold-by-well split、residual target
- 比較基準: exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960

## Feature 設計

- `sc8/sc15/sc25` score posterior、best code、margin、entropy
- PF / Beam / likelihood-PF / SC / hybrid / dense candidate の last-known 差分、likPF 差分、normalized gap
- candidate spread、range、bimodal midpoint、extreme gap、sign split、closest candidate gap
- prefix trust、`pfx_rmse`、known/eval length、near-row / longtail flags
- PF-Beam、Beam-likPF、SC15-likPF、dense-likPF gap の near/longtail interaction
- U-projection abs residual、slope、curvature、source disagreement

## 再現性設計

- seed policy: GroupKFold seed 42、LightGBM deterministic flags、fixed `n_jobs` / `num_threads`
- stochastic 処理の有無: 新規 feature generation には RNG なし。upstream PF/Beam cache と LightGBM GPU training は stochastic component として記録する。
- PF/Beam / likelihood-PF / seed bagging の有無: upstream exp072 / exp145 生成物を使う。current-test inference では exp148 と同じ replay generator を使う。
- 並列処理と乱数の関係: exp160 feature derivation は deterministic arithmetic のみ。PF/Beam raw-test replay は upstream generator の stable-seed policy に依存する。
- CPU/GPU runtime と deterministic flags: train active mode は `gpu_repro_guard_dp_threads8`。CPU control は config に残すが active にしない。
- train cache / test feature regeneration の SHA 記録方針: gzip は decompressed content SHA を主証拠にする。train summary / manifest に input schema と feature source を保存する。
- model manifest / prediction / submission SHA 記録方針: train では model SHA と OOF prediction SHA、inference では prediction SHA と submission SHA を保存する。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` 後に metadata と bootstrap 内 config の整合を確認する。

## リスク

- リークリスク: 公開 notebook の visible gold や true-error rank を入れると leak になるため、target-free columns だけを使う。
- CV/LB 不一致リスク: Public SP45 notebook に近い特徴を入れても Public LB 7.159/7.295 に近づいたとは解釈しない。ML route の exp148 改善として評価する。
- ランタイム/メモリリスク: exp148 full-row LightGBM に 50-100 features 程度を追加するため GPU runtime と memory が増える。
- 再現性リスク: GPU LightGBM と PF/Beam current-test replay は deterministic anchor ではない。採用判断には SHA と rerun evidence が必要。
