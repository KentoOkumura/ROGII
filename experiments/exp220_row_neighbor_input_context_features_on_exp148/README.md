# exp220_row_neighbor_input_context_features_on_exp148

## 目的

exp148 の learned-likelihood ML anchor に、同一 well 内の前後行から作る input/candidate/confidence context feature を add-only で追加し、row-local な連続性や不確実性の変化を LightGBM が利用できるか確認する。

## 状態

Kaggle CPU split train v1 完了。CPU 実行の timeout 対策として、学習 notebook は `train_lgb0` / `train_lgb1` / `train_lgb2` に分割した。control / parent の再学習は含めない。

## 背景

exp193 の `tlic_` context は train CV と LB を小改善した一方、exp180/190/161/166/172 などの大きな window/GR feature は不採用だった。今回は row-level context を少量に絞り、GR だけでなく PF/Beam delta、learned likelihood entropy/range、U-projection disagreement の近傍変化を見る。

## 仮説

同一 well 内の lag/lead/rolling context は、単独行特徴だけでは見えにくい「局所的に候補や信頼度が不安定な区間」を表現し、exp148 の residual model を小さく改善する可能性がある。

## 検証方針

GroupKFold 5 folds by well。active variant は `row_neighbor_input_context_addonly` のみ。LightGBM config は CPU で分割し、各 notebook は 1 config x 5 folds = 5 boosters を学習する。

比較基準は exp148 CPU Public LB 7.921、exp148 GPU `lgb_mean` CV 8.501281 / LB 7.960、exp193 `lgb_mean` CV 8.456665 / LB 7.946、exp198 `lgb_mean` CV 8.457924 / LB 7.930。

## 所見

3 split の OOF 予測を streaming aggregate した `lgb_mean` は RMSE 8.496282588。exp148 GPU `lgb_mean` 8.501281182 からは -0.004998594 改善したが、exp193 8.456665439、exp198 8.457923653、現行 ML submitted anchor の exp218 8.475793752 より弱い。

`rnic_` feature importance は `likpf_mean_d` と `uproj_source_u_std` の lead/lag 差が上位だったが、global CV の改善幅は小さい。exp220 は train-side completed / no submit とし、inference 化や提出には進めない。
