# exp342 exp226 Student-t residual-offset emission audit

## 状態

- Route: `pf_beam`
- 状態: Stage 1探索override完了、固定gate FAILで閉鎖
- 親実験: `exp281_exp226_residual_offset_exact_hmm_transition_probe`
- Stage 0 control: SHA固定済みexp280 Gaussian shift score
- Stage 1 control: SHA固定済みexp281 Gaussian HMM OOF
- inference / submission: 未実行

## 仮説

固定`df=4` Student-t emissionなら、大きなGR不一致が誤ったalignment modeを固定
する影響を弱め、exp281 Gaussian residual-offset HMMを改善できる可能性がある。

## 単一変更

Stage 1ではexp281の行別emission
`-0.5 * min(z^2, 600)`を`-2.5 * log1p(z^2 / 4)`へ置換しただけである。
offset grid、rate states、transition、prior、sigma、missing補間、exp226 path、
posterior meanは固定し、Gaussian親HMMは再実行していない。

## 検証方針

保存済みexp281 Gaussian OOFとのoverall/fold/1000+/hidden-like/by-well比較を行う。
Student-t候補をtruth join前にfreezeし、入力SHA、parent RMSE parity、finite、
row identityをhard guardする。scientific gateはexp281比`0.05 ft`、4/5 folds、
required scope非劣化、by-well p95非劣化、worst`<=+0.25 ft`のANDとし、
exp226以下のRMSEをdirect promotionの追加条件とする。

## 所見

Kaggle private CPU version 2（id_no `128356155`）を
`14,789.392992 sec`で完了した。Student-t HMMはGaussian exp281を
`9.827420 → 9.779772`へ`0.047648 ft`改善したが、必要な`0.05 ft`に届かなかった。
改善foldは3/5、hidden-like spatial / typewell-purgedは
`+0.014174 / +0.220136 ft`悪化、by-well p95は`+1.063793 ft`、
worst wellは`+12.893602 ft`だった。exp226 `9.427110`よりも
`0.352662 ft`悪く、scientific/direct promotionともFAILした。

Stage 0 proxyのFAIL後でも実HMMは少し改善したため、「Stage 0 FAILならHMMが絶対に
改善しない」という結果ではない。ただしfull HMM自身の固定gateでも改善の大きさ、
fold一貫性、hidden-like、tail safetyを満たさず、採用しない。

## 実装と実行量

- compact self-contained Jupytext train/inferenceと正規Notebookを実装済み
- Stage 0 version 1、Stage 1探索override version 2をKaggle CPUで完了
- Stage 1: 1 variant / 773 HMM well-runs
- LightGBM config / trained fold / booster / parent control再実行: `0 / 0 / 0 / 0`
- GPU / internet / inference / submission: すべて無効

## 判定

`stage_1_failed_close_without_rescue`として閉じる。df、scale、temperature、grid、
Huber、cap、missing、ACF、blendの救済、再実行、inference、submissionは行わない。
exp344 Huber依存条件も不成立のままとする。

## 次

追加実行は行わず、exp342固有の救済backlogも追加しない。

## 文書

- Steering: `../../docs/legacy/steering/20260722-exp342-exp226-student-t-residual-offset-emission-audit/`
- 設定: `config.yaml`
- 詳細結果: `result.md`
- 機械可読結果: `metrics.json`
