# exp247_missing_gr_masking

## 状態

実装・静的検証・Kaggle CPU train v1完了。`kentookumura/exp247-missing-gr-masking-train` version 1（id_no `127064272`）はCOMPLETEだが、一律maskは不採用として閉じた。推論と提出は行っていない。

## 仮説

exp221 exact HMM は horizontal GR の欠損を両方向補間した値まで観測として使う。特に長い missing run では人工的な GR 証拠が posterior を誤って引く可能性がある。raw GR 欠損 rowの GR emission contributionだけを0にし、transitionと固定exp148 LGB unaryで通過させると、欠損区間と直後の誤差を減らせるかを検証する。

## 検証方針

- 親: `exp221_lgb_oof_gaussian_emission_hmm_on_exp148`
- Route: `ensemble`
- 固定 control: exp221 train v3 `hmm_lgb_exp148_lgb_mean_s2000_l0500`
- 新規 variant: `mask_only` 1本
- LightGBM config / fold / booster: `0 / 0 / 0`
- parent/control再学習・再生成: なし
- 固定値: grammar、grid、transition、GR sigma、補間、LGB center、`sigma=20/lambda=0.50`
- 唯一の変更: evaluation suffix で raw horizontal GR が非有限のrowは、GR unaryを全stateで0にする
- 評価: overall、missing-run長、gap後128/256 rows、1000+、exp115 hidden-like、by-well、finite coverage、controlからの連続分岐長

visible testはGR欠損分布の記述にだけ使い、score evidenceやgateには使わない。true TVT、OOF error、hidden-like role、oracleはmask生成へ渡さない。

## 所見

overall RMSEは8.327728213 -> 8.322894658（-0.004833555 ft）と微改善した一方、MAEは+0.042731938 ft悪化した。短missing run 1-31もRMSE -0.003848864 ftに対しMAE +0.067176896 ft、hidden-like spatialはRMSE +0.005961600 ft、worst wellは+2.576980644 ft悪化した。well別は改善386 / 悪化387で、finite coverageは両候補100%だった。小さく一貫しないaggregate gainと大きなtail regressionのため、一律mask、run-length gate、inference、submitは不採用とする。

## 生成物

- raw train/test missing well / run inventory
- fixed exp221 control missingness readout
- mask-only prediction cacheとrow audit
- overall / missing / post-gap / distance / hidden-like / by-well metrics
- controlからのdivergence segmentとfinite coverage
- input/output SHAを含むsummary

## 禁止事項

初回結果からrun-length gate、raw-test inference、selector、submissionへ自動的に進まない。短いmissing-run bucketを壊した場合もthreshold gridを続けず、一度この方向を閉じる。

## 次のアクション

このbranchは完了・不採用として閉じ、派生実験は追加しない。
