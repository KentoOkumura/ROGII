# exp242_two_regime_rate_noise_pf 結果

## 状態

Kaggle train version 2でfull train-side auditを完了し、不採用としました。
raw-test inferenceとsubmissionには進みません。

## 実装した仮説

exp072-compatible likelihood-PFへstickyな`smooth / turn`状態を追加し、`turn`時だけrate
process noiseを4倍にしました。観測尤度、position noise、resampling、particle/seed数は固定です。

## 親実験との差分

exp072-compatible `(position, rate)` stateへ`regime`だけを追加しました。保存済みexp072
`likpf_mean`を比較controlとして再利用し、control PFは再生成していません。

## 実行結果

- 対象: 3,783,989 rows / 773 wells、coverage 1.0。
- runtime: 23,665.002秒（約6時間34分25秒）、Kaggle CPU。
- exp072 `likpf_mean`: RMSE 11.594898、MAE 7.067633、within10 0.772807。
- `pf_two_regime_k4_mean`: RMSE 13.254455、MAE 8.912314、within10 0.678840。
- RMSE差は+1.659557で明確に悪化した。
- 全distance bucketで悪化し、`1000_plus`は12.704015から14.457482（+1.753467）。
- hidden-like spatialは13.643808から14.508022（+0.864215）、typewell-purgedは
  13.506801から14.412865（+0.906064）。
- well単位では275 wells改善、498 wells悪化、median deltaは+1.236399。
- 最大well回帰は`c8d9680c`の+41.956968。

## Regime診断

- turn particle fraction平均: 0.018088。
- turn posterior mass平均: 0.017897。
- entry / exit / switch fraction平均: 0.000196 / 0.000365 / 0.000562。
- posterior massがparticle fractionを上回らず、GR likelihoodが高noise turn regimeを
  平均的に支持した証拠はない。resampling後もturn粒子の拡散が予測精度へ結びつかなかった。

## 採否

固定sticky regimeとturn時4倍rate noiseは不採用です。overall、long-tail、hidden-like、
worst-wellの全guardに失敗したため、transition、初期比率、multiplierの追加gridは行いません。
raw-test inferenceとsubmissionも行いません。

## 再現性と生成物

- Kaggle kernel: `kentookumura/exp242-two-regime-rate-noise-pf-train` version 2。
- exp072 validation source decompressed SHA: `99a3c70a...320e1350`。
- exp209 reconstructed control decompressed SHA: `ee3b548b...d2ee3f4`。
- row candidates decompressed SHA: `13ca093b...b448635`。
- metrics、distance、hidden-like、by-well、PF/regime診断、row candidates、summaryを保存した。

## 次のアクション

dynamic high-noise regimeの再調整は止めます。次のPF/HMM系候補では、既存backlogの
`multi_scale_initial_rate_candidates`のように、known prefixだけから作る離散的な初期rate候補を
baseと並存させ、tail中にprocess noiseを継続注入しない設計を優先します。
