# exp311_typewell_group_prefix_suffix_gr_calibration_readout

## 状態

- ルート: `pf_beam`
- 状態: Kaggle private CPU version 1完了・固定gate FAIL・branch closed
- 親: `exp211_affine_calibrated_gr_observation_pfbeam`
- CV / LB: train-side readout完了 / 未提出

## 仮説と変更点

同じ`native_overlap_1` Type Well群では、GRのhard affine係数より残差noise/reliabilityがwellをまたいで共有される。outer-train TVT truthだけから群統計を作り、held-out suffixへの転送性を0-boosterで監査する。予測・GR補正・decoderは変更しない。

## 検証方針

SHA256 well-grouped 5-fold、leave-one-group-out、spatial/typewell-purgedを用い、truthはreal/control priorのSHA凍結後だけscoreへ結合する。primaryは`native_overlap_1`、`exact_typewell_content_sha`は感度分析とする。group LOO、suffix horizontal-GR RMSE gain、negative control、4/5 folds、worst-well guardを全PASSするまで後続利用不可。

## 所見

Kaggle version 1ではprimary `native_overlap_1` のsame-group held-out-wellでidentity比GR-RMSE gain `0.376220`、5/5 folds改善、group-shuffle差 `0.240055`、noise R² `0.202320`を得た。一方fit-RMSE R²は`-0.003255`、worst-well GR-RMSE deltaは`+12.914716`で固定gateを2件失敗した。平均的なnoise転送性はあるが、fit品質の群間説明力とtail safetyを満たさないためpromotionしない。

## 実行入口

compact trainを正規train notebookへ採用し、`kentookumura/exp311-typewell-gr-calibration-readout-train` version 1としてprivate CPU / internet offで実行した。実行契約は1 diagnostic / 5 folds / 0 model / 0 booster / 0 decoder。inference/submissionはfail-closedのまま実行していない。

## 次

同じOOFでgateやshrinkageを救済調整せずbranchを閉じる。exp311 PASSを先行条件としたexp312〜320は停止し、独立した事前根拠がない限り実装・実行しない。
