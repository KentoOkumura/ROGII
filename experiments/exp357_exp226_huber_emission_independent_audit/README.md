# exp357 独立 Huber emission 監査

## 状態

- Route: `pf_beam`
- 状態: Stage 1 exact-HMM完了、guard FAILで救済なし終了
- 優先度: 低・P4
- 親: `exp281_exp226_residual_offset_exact_hmm_transition_probe`
- 保存control: `exp280_exp226_shift_likelihood_separability_readout`
- Kaggle: private CPU version 2 / id_no `128448451`

## 仮説

center付近のGaussian曲率を保ち、extreme residualだけをlinear tailにする固定Huberなら、
Gaussian shift rankを安全に改善できるかを独立監査した。

## 検証方針

exp280 Gaussian control、512-row block、13 shifts、exp281 sigmaを固定し、
pooled MRR/top3各`+0.01`、4/5 folds、stress、circular、extreme-residualを
AND gateで評価した。score bundleはtruth join前にSHA固定した。

## 結果

fixed `delta=1.345` HuberのGaussian比pooled gainはMRR `+0.0000416`、
top3 `+0.0001284`で、固定下限の各`+0.01`に届かなかった。改善foldも両方`2/5`、
stress MRR/top3非悪化もFAILした。

一方、174 extreme-residual blocksではtop3 `+0.0114943`、regret
`-0.652270 ft`で改善した。技術control、truth-late join、real-vs-circularはPASSしており、
実装不良ではなく、tail局所改善が全体・stressへ転化しないnegative resultと判断する。

## 所見

robust tailによるextreme-residual局所改善は確認できたが、全体rank gainは実質的に
Gaussian parityで、marginのflatteningとstress回帰を伴った。fixed Huberを
exp281 exact HMMへ昇格させる科学的根拠は成立しない。

## 実行契約

- scientific score / saved control: `1 / 1`
- shift candidates / reporting folds: `13 / 5`
- HMM well-run / model config / trained fold / booster: `0 / 0 / 0 / 0`
- parent/control再学習、inference、submission: すべて`0`

## Stage 1 override

ユーザー明示依頼により、Stage 0の通常停止条件をoverrideしてfixed Huber
exact-HMMを1 variant / 773 wellsで評価する。exp281の状態空間・遷移・sigma・
missing policy・出力は固定し、Gaussian行別emissionだけをHuberへ置換する。
親control再実行、学習model、booster、inference、submissionはすべて0。

実HMMのRMSEはGaussian `9.827420`からHuber `9.737195`へ`0.090225 ft`改善し、
4/5 folds、1000+、hidden-like 2面も改善した。一方、by-well p95 deltaは
`+0.003365 ft`、worst well `4a8ecc0b`は`+1.403715 ft`で固定安全gateをFAILした。
exp226 direct ceiling `9.427110`にも`+0.310086 ft`劣るため、promotionはFAILとする。

## 結論

Stage 0は`stage_0_failed_close_without_rescue`、Stage 1は
`stage_1_failed_close_without_rescue`。Stage 0 proxyに反して実HMMの平均改善は
確認できたが、well-level tailとdirect ceilingが不十分で採用できない。
delta/scale grid、Student-t同時評価、sigma/tempering/blendによる救済、
inference、submissionは行わない。
