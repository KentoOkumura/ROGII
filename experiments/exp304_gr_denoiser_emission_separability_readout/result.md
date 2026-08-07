# exp304_gr_denoiser_emission_separability_readout 結果

## 結論

Kaggle private CPU version 1はtechnical gateとquality gateをPASSした。固定3方式のうち、
`swt_db4_l3`だけが全1,546 horizontal/typewell seriesでtechnical PASSし、事前登録した全quality gateを通過した。
`selected_denoiser`は`stationary db4 level-3 SWT`に確定する。

この結果はGR emission separabilityの診断結果であり、CV RMSE、HMM/PF/Beam性能、Public LBの改善を意味しない。
inferenceとsubmissionは生成していない。

## 実行

- Kaggle kernel: `kentookumura/exp304-gr-denoiser-separability-train`, version 1, id_no `128011752`
- 状態: `COMPLETE`
- runtime: `4,740.758 sec`（約79.0分）
- surface: 3,783,989 rows、773 wells、7,787 blocks、13 shifts、保存済み5 fold strata
- 計算量: 4 readout variants、model / LightGBM config / trained fold / booster / HMM / PF / Beamは全て0

## 結果表

| variant | technical | MRR | top1 | top3 | raw比MRR | raw比top3 | quality gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| raw | PASS | 0.389626 | 0.189547 | 0.452421 | - | - | control |
| robust_rts | FAIL | - | - | - | - | - | 非評価 |
| swt_db4_l3 | PASS | 0.424724 | 0.220624 | 0.504687 | +0.035098 | +0.052267 | PASS / selected |
| l1_trend | FAIL | - | - | - | - | - | 非評価 |

SWTはmean rankも`4.653140 -> 4.195582`へ改善し、truth-minus-best-decoy gapは
`-0.078518 -> -0.042601`（`+0.035917`）へ改善した。MRR/top3は5/5 foldsでrawを上回り、
real-vs-shuffledも5/5 foldsでPASSした。`md_since_1000_plus`、hidden-like spatial、
hidden-like typewell-purged、sharp-edgeの全必須scopeでMRR/top3非悪化を満たした。

## Technical gate

- raw: 1,546/1,546 series PASS、7,787 blocks / 101,231 candidate rows
- SWT: 1,546/1,546 series PASS、7,787 blocks / 101,231 candidate rows、PyWavelets 1.8.0
- robust RTS: 15/1,546 seriesだけが最大8反復内に収束し、1,531 failures。technical FAILのためquality非評価
- L1 trend: 572/1,546 seriesが最大500 ADMM反復内に収束し、974 failures。768 blocksだけの部分scoreなのでquality非評価
- silent fallback: 0。失敗方式を別方式や低いlevelへ置き換えていない
- common freeze / row-block-fold identity / finite coverage: PASS

RTS/L1の反復数や収束閾値を同じOOFで変更する救済gridは行わない。

## 再現性証拠

- exp226 OOF decompressed SHA: `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`
- exp115 hidden-like SHA: `5f9ac9fac6bb3725a7c613f09856a85bdf73b8206fd2edf1b79e8eaa9bca6597`
- raw well-file identity content SHA: `bbb687a1998092578583ce259309b49031d095bde57cbb26c0ab8808d2379b32`
- scientific contract content SHA: `8822df968200b74ea9969b0bc023ec127debbff01933bdc89ff3db9844d55064`
- denoised GR content SHA: `a4acb72d60b833b12b2560db1e5dc3a113ae6ecf4137efbccf78c278582a0988`
- target-free score content SHA: `6c71bdae030fee04e40988a8abdef4e26b61af733c463b6b0625e2e0cc99fa69`
- block readout raw/decompressed SHA: `c312305820c2c8ebf7f5574090b2fb93dc3e8cc769f6c2c60033cae28cf3e9b8` /
  `239c990260032d896894d0903ed99e7cbb39796e1e6357218acfe4869a9f6623`（15,574 data rows）

小型summary/manifest/metrics/solver/fold/scope生成物はKaggle記録SHAとローカル取得後SHAが全件一致した。
大容量denoised seriesは一括取得と単独file-pattern再取得の両方でCLIが0 byteを返したため、Kaggle側の
output存在、manifestの6,659,300 rows、完了ログのraw/decompressed/content SHAを証拠とする。
大容量本体はリポジトリへ保存せず、案2ではkernel source上のnonzero sizeとmanifest SHA一致を実行前hard preflightにする。

## 判断

exp304は`train_side_readout_completed_quality_passed`として完了する。予約契約の案2
`tempered_raw_smoothed_exact_hmm_emission`は開始条件を満たしたため、別実験として設計可能である。
SWT選択なのでRTS posterior varianceを前提とする案3は閉じる。案4は案2がPASSするまで開始しない。

案2でもraw emissionを捨てず、`0.85 * ell_raw + 0.15 * ell_swt`の固定mixtureだけを評価する。
同じexp304でHMM/PFへ進めず、filter/beta/sigma/clip/transitionの救済grid、inference、submissionは行わない。
