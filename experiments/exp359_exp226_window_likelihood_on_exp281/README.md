# exp359 exp226 window likelihood on exp281

## 状態

- Route: `pf_beam`
- 状態: Stage 0完了・固定gate FAIL・救済なし閉鎖
- 優先度: 低・P4
- 親: `exp281_exp226_residual_offset_exact_hmm_transition_probe`
- Stage 0 control: `exp280_exp226_shift_likelihood_separability_readout`
- 履歴参照: `exp325_exp226_window_likelihood_hmm_tempering`

## 仮説

exp226の500-row GR window evidenceをfixed stride中心だけへ正規化して加えると、
exp281のrow-wise Gaussian emissionでは分離できないresidual-offset modeを
識別できる可能性がある。

## 実装

- full 500-row window / stride 125 / 固定13 shifts
- exp226 affine-calibrated GR profileのcorrelation/MSE/level固定score
- state方向標準化と13-shift softmax posterior SD
- `125/500 * clip(1.1-0.12*sd,0.3,1.0)`の固定lambda
- well/window/profile SHA由来の決定的score permutation
- window centerを含む保存済みexp280 512-row blockへのcontrol lookup
- window/profile/eligibility/lambda/shuffle/control bundleのSHA固定後だけtruth join

compact self-contained train候補を正規`*_train.ipynb`へ採用した。
正規`*_inference.ipynb`はplaceholderのまま保持し、別名のfail-closed
inference候補だけを作成している。

## 親からの変更点

exp281のrow Gaussian emission、transition、sigma、grid、momentum、prior、
posterior outputは変更しない。Stage 0で追加するのはfixed sparse window
potentialのrank readoutだけで、exp226補正値やfinal predictionは使わない。

## 検証方針

Stage 0はsaved exp280 control比MRR/top3各`+0.01`、各4/5 folds、
real>shuffle 5/5 folds、1000+、hidden-like 2面の正方向、
eligible window 25%以上をAND gateにする。実行量はscientific score 1 /
saved control 1 / reporting fold 5 / HMM・model・trained fold・booster各0。

## 実行

- 実装候補:
  `exp359_exp226_window_likelihood_on_exp281_compact_selfcontained_train.ipynb`
- inference停止候補:
  `exp359_exp226_window_likelihood_on_exp281_compact_selfcontained_inference.ipynb`
- canonical private CPU kernel:
  `kentookumura/exp359-exp226-window-likelihood-on-exp281-train`
- version / id_no: `1 / 128528648`
- runtime: `4523.211267 sec`
- Stage 0実行量: scientific score 1 / saved control 1 / 5 folds /
  HMM・model・trained fold・booster各0

## 現時点の結果

技術gateはcoverage/parity各1.0、eligible window fraction `0.385617`、
real>shuffle `5/5 folds`でPASSした。

科学gateはFAIL:

- window / control MRR: `0.372904 / 0.395168`（差`-0.022264`）
- window / control top3: `0.414471 / 0.447968`（差`-0.033496`）
- MRR / top3改善fold: `0 / 5`, `0 / 5`
- long-tail 1000+、hidden-like spatial、typewell-purgedの全scopeで負方向
- 全10,628 eligible windowsでlambdaが下限`0.075`へ飽和

## 所見

長いwindow profileはshuffleより明確に強いが、saved row-Gaussian aggregateより
全fold・全stress scopeで弱い。同じOOFを見たwindow/stride/weight/lambda gridは
行わず、exp281 HMMへ追加する根拠なしと判断する。

## 次のアクション

Stage 1、inference、submissionへ進まず、exp359を閉じる。
