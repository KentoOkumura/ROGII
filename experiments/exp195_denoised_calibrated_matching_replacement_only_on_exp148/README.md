# exp195_denoised_calibrated_matching_replacement_only_on_exp148

## 目的

exp148 の `learned_likelihood_confidence` block を、target-free な denoised calibrated GR matching feature block へ置き換え、exp190 add-only で見えた `lgb1` 単体改善が full block replacement で安定するかを確認する。

## 状態

Kaggle train v1 完了。train-side rejected。inference / submit は行わない。

## 背景

- exp190 add-only は `lgb_mean` 8.503596159 で exp148 8.501281182 から +0.002314978 悪化した。
- 一方で exp190 `lgb1` は exp148 同 config から -0.024346641 改善した。
- add-only では exp145 learned likelihood confidence と DCM signal が競合した可能性があるため、今回は `ll_*` block を外す replacement-only として切り分ける。

## 仮説

GR shift-scan surface の sharpness / ambiguity / smoothing gain は、exp145 learned likelihood confidence と同じ「候補信頼度」役割を持つ。両方を同時投入するのではなく、`learned_likelihood_confidence` を完全に外して DCM block に置き換えると、重複 signal の競合が減り、exp148 anchor を改善する可能性がある。

## 変更点

- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- route: `ml_model`
- variant: `denoised_calibrated_matching_replacement_only`
- active feature groups: `projection_correction`, `u_disagreement`, `denoised_calibrated_matching`
- removed feature group: `learned_likelihood_confidence` (`ll_*` 54列)
- control: 再学習しない。保存済み exp148 CV / Public LB を historical baseline として参照する。

直接 TVT 置換、blend、postprocess、hard selector、heel calibration、FFT notch は入れない。

## 検証方針

GroupKFold 5 folds を well group で実行し、GPU LightGBM family 3 configs を単一 train notebook 内で学習する。予定は 1 variant、3 configs、5 folds、合計 15 boosters。

比較基準:

- exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960
- exp190 add-only `lgb_mean` CV 8.503596159484825
- exp160 `lgb_mean` CV 8.463718773783008 / Public LB 8.061
- exp162 rank-slot split CV positive / LB negative

train CV が positive でも、raw-test/current-test feature parity が未実装のため、そのまま submit しない。

## 所見

実装段階では未評価。Kaggle train 後に pooled OOF、fold別 score、feature importance、near `000_050`、`1000_plus`、worst-well、PF-dense disagreement bucket、exp115 hidden-like subgroup を確認する。
