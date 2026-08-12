# 設計

## アプローチ

`exp043` をコピーし、同じ feature-family 生成と同じ split surface を維持する。
変更点は、候補の成功条件と postprocess 候補の出し方に限定する。

## 実験範囲

- 対象実験: `exp048_ravaghi_single_model_feature_parity_revisit`
- Route: `ml_model`
- 親実験: `exp043_ravaghi_feature_family_ablation_matrix`
- 変更する変数:
  - support 判定を direct PF controls 超えに変更する。
  - raw / bucket shrink / anchor gate / public PF blend を候補として保存する。
  - feature parity report を追加する。
- 固定する変数:
  - 入力 feature artifact。
  - original-fold / well-hash split。
  - LightGBM の主要パラメータ。
  - exact beam / NCC / GR match の fold-safe 再生成方針。

## リスク

- リークリスク: cutoff 以降の true TVT を特徴生成に使うと成立しない。既存コードの masked `TVT_input` 再生成を維持し、target は score のみに使う。
- CV/LB 不一致リスク: train 側の見えない test 風 surrogate と Public LB が一致しない実例がある。direct PF controls 超えを必須条件にし、full audit 後も即 inference port しない。
- ランタイム/メモリリスク: exp043 は Kaggle train で約 3,900 秒かかった。postprocess 候補は学習を増やさないが出力候補数が増えるため、CSV サイズと notebook runtime を確認する。
