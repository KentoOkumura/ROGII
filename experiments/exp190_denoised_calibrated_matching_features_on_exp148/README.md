# exp190_denoised_calibrated_matching_features_on_exp148

## 目的

exp148 の learned-likelihood ML anchor に、target-free な GR shift-scan confidence feature を add-only で追加し、raw/smoothed matching surface の sharpness と posterior ambiguity が LightGBM に効くかを確認する。

## 状態

Kaggle train v1 完了。train-side CV では exp148 を超えなかったため、不採用。inference / submit は行わない。

## 背景

- exp167: FFT notch は弱いが、rolling median / Savitzky-Golay smoothing は gap / entropy / decoy gap を改善した。
- exp170: heel calibration は shift-scan / PF observation rank を悪化させたため採用しない。
- exp171: bimodal posterior candidate の direct replacement は不採用。posterior は confidence / ambiguity feature としてのみ扱う。

## 仮説

GR shift-scan surface の sharpness / ambiguity / smoothing gain は、exp148 の learned likelihood confidence とは別系統の不確実性信号になり、LightGBM が PF/Beam/likPF candidate を信用すべき行を判別する補助特徴として効く可能性がある。

## 変更点

- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- route: `ml_model`
- variant: `denoised_calibrated_matching_addonly`
- control: 再学習しない。保存済み exp148 CV / Public LB を historical baseline として参照する。

直接 TVT 置換、blend、postprocess、hard selector、heel calibration、FFT notch は入れない。

## 検証方針

GroupKFold 5 folds を well group で実行し、GPU LightGBM family 3 configs を単一 train notebook 内で学習する。予定は 1 variant、3 configs、5 folds、合計 15 boosters。

比較基準:

- exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960
- exp160 `lgb_mean` CV 8.463718773783008 / Public LB 8.061
- exp167 / exp171 の matching audit 所見

train CV が positive でも、raw-test/current-test feature parity が未実装のため、そのまま submit しない。

## 所見

Kaggle train v1 は 3,783,989 rows / 773 wells / 431 features / 15 boosters で完了し、feature join coverage は full pass。

pooled OOF は `lgb0` 8.601678275、`lgb1` 8.539624480、`lgb2` 8.540073562、`lgb_mean` 8.503596159。`lgb1` は exp148 同 config から -0.024346641 改善したが、採用基準の `lgb_mean` は exp148 `lgb_mean` 8.501281182 から +0.002314978 悪化した。

したがって denoised calibrated matching feature block は exp148 の add-only feature として採用せず、current-test parity 実装、inference、submit には進めない。
