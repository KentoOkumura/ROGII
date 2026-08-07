# exp309_well_adaptive_transition_noise

## 状態

- Route: `pf_beam`
- 状態: upstream exp307 chain FAILにより未実行のまま閉鎖
- 親: `exp308_imputed_gr_confidence_downweight`

## 仮説

known prefixの`U=TVT+Z` rate変動からwell別`sig_r`を作れば、全well固定0.002より適切な探索幅になる。

## 実装

- rate差のMADを`n/(n+100)`で0.002へlog shrinkする。
- 20 innovation未満は0.002、clipは`[0.001,0.004]`とする。
- exp307 finite-MAD `σ_GR`とexp308 missing-distance confidenceを固定実装した。
- `sig_p`、41 rate states、momentum、GR補間、typewell、prior、posterior meanは固定した。
- 1 variant、773 HMM runs、0 booster、parent/control再実行0とする。
- exp308 promotion status、prediction SHA、parent metricsが揃うまで実行入口で停止する。

## 検証方針

parentより0.05 ft以上・4/5 folds改善し、fallback/clip率、1000+、hidden-like、p95、worst、fixed blendを守る。sig-r/support/turning/distance別readoutも保存する。

## 結果

コード、train/inference Notebook、契約テストを実装済み。CV、LB、prediction、submissionはなく、Kaggle package/push/runも行っていない。

## 所見

固定仕様を実行可能な形にはしたが、精度に関する証拠はまだない。prefix rate diffusionがsuffixへ転送できるかは、exp308 promotion後のKaggle train-side auditでのみ判定する。

## 次

exp308が未実行のまま閉じたため、Kaggle実行、inference、submissionへ進まない。
