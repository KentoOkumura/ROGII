# exp280_exp226_shift_likelihood_separability_readout 結果

## 状態

Kaggle private CPU kernel version 1でfull readoutを完了し、固定separability guardはPASSした。
これは予測CVではなくtrain-side diagnosticであり、CV / LBはない。inference / submissionは
設計どおり実行していない。

## 仮説

exp226 `tvt_geop`の局所形状が正しく、誤差の主成分が縦offsetなら、raw GR/typewell likelihoodは
固定shift bank内のtruth-nearest候補を5 foldsで一貫してrankできる。

## 設定

- 親: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- 失敗参照: `exp279_exp226_geop_centered_exact_hmm_redecode`
- 検証: exp226 group-safe OOF、非重複512行block、末尾short block保持
- shift: `[-80,-40,-20,-10,-5,-2,0,2,5,10,20,40,80] ft`
- likelihood: exp209 Gaussian raw-GR/typewell emission、known-prefix sigma clip 10-60
- メトリック: top1 / top3 / MRR / sign、real vs stable shuffled、coverage/margin/regret
- シード: 42
- 実行量: 1 audit variant / 0 config / 0 trained fold / 0 booster / 0 HMM

## 結果

| メトリック | 値 |
| --- | --- |
| guard | PASS |
| top1 | 0.189547 vs shuffled 0.075767（+0.113779） |
| top3 | 0.452421 vs shuffled 0.234493（+0.217927） |
| MRR | 0.389626 vs shuffled 0.245536（+0.144090） |
| sign | 0.498523 vs shuffled 0.418518（+0.080005） |
| mean rank | 4.653140 / 13 candidates |
| top1 regret RMSE | mean 13.955240 ft / p90 38.615667 ft |
| guard fold count | top1/top3/MRR/signすべて5/5 |
| coverage | row identity / finite score / bank range / quantization = 1.0 / 1.0 / 1.0 / 1.0 |
| 対象 | 3,783,989 rows / 773 wells / 7,787 blocks |
| runtime | 456.972秒 |
| Public LB | - |
| Private LB | - |

fold別liftはtop1が`+0.101440～+0.125935`、top3が`+0.186518～+0.246523`、
MRRが`+0.128698～+0.159668`、signが`+0.064136～+0.099751`で、全foldが正だった。

主要scopeでも1000+はtop1/top3/MRR/sign liftが
`+0.097244 / +0.195771 / +0.128153 / +0.071451`、hidden-like spatialは
`+0.093953 / +0.196902 / +0.125365 / +0.089455`、persistent-offsetは
`+0.071279 / +0.155556 / +0.099719 / +0.067925`で全て正だった。nearは1 blockだけなので解釈しない。

## 再現性

- deterministic anchor: いいえ。prediction/submissionを作らないdiagnostic。
- seed policy: real scoreはRNGなし、shuffleだけstable SHA256 local RNG。
- Kaggle kernel: `kentookumura/exp280-exp226-shift-likelihood-readout-train` version 1、
  id_no `127828902`、private CPU、GPU/TPU/internet off。
- pushed config SHA: `4a95c5143f8decd163be14913d5bf4717f76513f724e2655e2eaa8523ff70ed1`。
- train source SHA: `2f6666e1602e99270f725c080546288fb2b8615b3a392a4fd1bf1a2970bd1db3`。
- scientific contract SHA: `60d32ba96e0f71fc1f02f53d9e274e97d96516549ee27c87ea40bbb666af7978`。
- input SHA: exp226 decompressed `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`、
  hidden-like raw `5f9ac9fac6bb3725a7c613f09856a85bdf73b8206fd2edf1b79e8eaa9bca6597`。
- target-free scoreはcontent SHA
  `4a546cfe5f9291168bdb4dcb912182b079e0343af845f76005f6a7100ac3aa46`を確定後にだけtruthを結合。
- target-free score gzipのraw / decompressed SHAは
  `fd698c81...e2bad / c6e9e39a...d99c3`、block readoutは
  `fe2b6527...d48b8 / c1cd8fb1...16ee3`でKaggle summaryと取得ファイルが一致した。
- model / prediction / submission SHA: 対象外。

## 解釈

raw GR/typewell likelihoodには、exp226 geometry周囲のtruth-nearest offset候補をshuffledより
良く順位付けするfold-stableな情報がある。したがって、exp226座標系でoffsetだけを状態にする
後続HMMの先行条件は満たした。

一方、top1は18.95%、persistent-offsetでは15.05%、sign絶対精度は49.85%に留まる。
この結果はhard shift correctionを支持せず、時間方向のslow-offset grammarで弱いlikelihoodを
統合する検討だけを支持する。near scopeは標本不足であり、後続実験ではnear非悪化guardを維持する。

## 次

`exp226_residual_offset_exact_hmm_transition_probe`を別実験として設計する。exp209 emissionと
missing処理を固定し、`TVT_t = exp226_geop_t + delta_t`のslow offsetだけを1 fixed grammarでdecodeする。
exp280のshift/grid/calibrationを救済探索せず、direct correction、inference、submissionには進めない。
