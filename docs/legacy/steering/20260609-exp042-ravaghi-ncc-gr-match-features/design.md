# 設計

## 仮説

Ravaghi multi-scale NCC / GR match confidence は、raw GR 値を直接入れなくても
PF/beam candidate の信頼度や disagreement を表す診断特徴として single LightGBM
に効く可能性がある。

## アプローチ

`exp041` の single-LGBM ablation framework を再利用する。`exp029` train well の途中以降を隠した疑似 test
feature CSV を読み込み、必要な base/control columns だけを保持した後、各
`(well_id, cutoff_row)` について train horizontal/typewell CSV から Ravaghi NCC/GR
match features を再生成する。

NCC は Ravaghi と同じ `half_windows=(8,15,25)`、`stride=3`、score softmax ensemble
を使う。model feature には path delta、score/confidence、PF/beam disagreement、
typewell GR residual offset を入れる。raw GR 値は入れない。

## 実験範囲

- 対象実験: `exp042_ravaghi_ncc_gr_match_features`
- Route: `ml_model`
- 親実験: `exp041_ravaghi_beam_exact_feature_ablation`
- 変更する変数: Ravaghi NCC/GR match feature families と variants
- 固定する変数: exp029 input artifact、LightGBM params、postprocess bucket shrink、original-fold / well-hash audit surfaces、report controls

## リスク

- リークリスク: pseudo cutoff 後の true `TVT` や train-only formation columns を使わない。`target_tvt` は label/scoring 専用にする。
- CV/LB 不一致リスク: GR alignment 系は exp008/exp017 で悪化済み。Ravaghi 実装に寄せた小さい family として 1 回だけ検証し、direct PF controls と分けて評価する。
- ランタイム/メモリリスク: NCC は `(well_id, cutoff_row)` 単位で生成するため exact beam より軽い想定だが、full 773 wells では Kaggle train runtime を記録する。

## 次のアクション

Kaggle train version 1 の結果、NCC/GR match は base geometry より改善したが direct PF controls と exp041 exact beam disagreement より弱かった。inference port / submit は行わず、今後は family matrix の診断値として扱う。
