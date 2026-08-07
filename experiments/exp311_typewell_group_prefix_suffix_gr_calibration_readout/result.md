# exp311 結果

## 状態

Kaggle private CPU version 1を完了し、固定promotion gateはFAILした。実行は正常終了し、1 diagnostic / 5 folds / 0 model / 0 booster / 0 decoderの契約、late-truth境界、10生成物、SHAを確認した。inferenceとsubmissionは実行していない。

## 仮説と変更点

同じType Well群のbias/noise/reliabilityがheld-out wellへ転送可能かを、hard補正やdecoderを使わずsuffix horizontal-GR再構成で監査する。設計-only scaffoldから実行可能なreadoutへ変更した。

## 固定した判定

- 主評価: group LOO R²、held-out suffix horizontal-GR RMSE gain、negative-control差。
- promotion: noise/fit R²≥0.20、GR gain≥0.05、4/5 folds、shuffle差≥0.03、worst GR-RMSE delta≤+0.25。
- FAIL時: hard affine correction、HMM/PF/Beam、下流exp312〜320へ進まない。

`ft`単位の評価にはTVT候補またはdecoderが必要で本実験の禁止事項に反するため、suffix reconstructionはhorizontal GR API unitで評価する。TVT予測、GR値の書換え、候補生成は行わない。

## Kaggle version 1結果

- kernel: `kentookumura/exp311-typewell-gr-calibration-readout-train` version 1、id_no `128085784`。
- runtime: `246.631 sec`、private CPU、internet off。
- primary: `native_overlap_1` / `same_typewell_heldout_well`。
- coverage: 760/773 available wells（98.318%）、773 scored wells。
- pooled identity GR-RMSE `11.745716` → transfer `11.369495`、gain `0.376220`。
- fold gain: `0.668450 / 0.449965 / 0.389351 / 0.133892 / 0.230035`、5/5 folds改善。
- shuffled pooled gain `0.136165`に対するreal差 `0.240055`。
- noise R² `0.202320`（PASS）、fit-RMSE R² `-0.003255`（FAIL）。configの`group_loo_*` keyは固定済みの旧名称で、実装は`primary_surface`のsame-group held-out-well R²を読んでいる。
- worst-well GR-RMSE delta `+12.914716`（許容`+0.25`、FAIL）。
- spatial/typewell-purged pooled gain `0.349090`、4/5 folds改善だがworst delta `+12.578262`。
- exact-hash感度分析は28/773 wells（3.622%）しか利用可能で、pooled gain `0.013695`だった。

8 gateのうち6件はPASSし、失敗はfit-RMSE R²とworst-well safetyの2件。群noiseには弱い転送性が見えるが、群priorの品質指標と個別well安全性が再現せず、下流利用条件を満たさない。

## 生成物

Kaggle outputの9 CSV/CSV.GZとsummary JSONを一時領域へ取得した。全9 manifestのraw SHAが一致し、pair table 3,579,906 rowsのgzip展開後SHA `14f506da542a0d6f460425ddb56ff7119e19699b7d6110956fde86e63311e335`も一致した。summary JSON SHAは`821499b895a8bdb6ed8202c714ddb95ea8739defe9b4aaa0e13c35b346a1ca29`。大容量生成物はGit管理下へ保存していない。

## 次

同一結果で閾値、shrinkage、group定義を調整しない。exp311全gate PASSを前提とするexp312〜320は停止する。次の優先順位は独立した既存P0のexp321 Stage A/Bと、exp304 PASSから続くexp305を維持し、Type Well群transfer branchは新しい独立根拠が得られるまで再開しない。
