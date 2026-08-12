# 設計

## 仮説

exp226の500-row GR window evidenceは、exp281のrow-wise Gaussian emissionだけでは
区別しにくいresidual-offset modeを固定13-shift上で分離できる。

## アプローチ

exp281 residual-offset HMMを固定し、exp226の500-row GR correlation/MSE/level scoreを
state shiftごとに計算して、stride 125のwindow中心へだけsparse factorとして追加する。
exp226 prediction/correction pathは使わず、window evidenceの識別力をStage 0で先に測る。

旧exp325のfixed formulaを維持するが、exp323 transition scheduleは削除し、
exp281保存surfaceへ直接接続する。exp281自体のtail失敗を踏まえ、Stage 1には
exp226単体以下というabsolute ceilingを設ける。

## 実験範囲

- 対象実験: `exp359_exp226_window_likelihood_on_exp281`
- Route: `pf_beam`
- 親実験: `exp281_exp226_residual_offset_exact_hmm_transition_probe`
- 変更する変数: sparse exp226 window potentialだけ
- 固定する変数: exp281 row emission、transition、sigma、grid、momentum、prior、output
- window:
  - rows 500 / stride 125
  - shift grid `[-80,-40,-20,-10,-5,-2,0,2,5,10,20,40,80]`
  - minimum finite GR fraction 0.50 / minimum profile bins 7
  - weights correlation 2.0 / MSE 0.5 / level 0.1
  - state mean subtraction + state std floor `1e-6`
  - lambda `stride/window * clip(1.1-0.12*posterior_sd,0.3,1.0)`
- Stage 0: diagnostic 1 / saved Gaussian aggregate control 1 / HMM・model・booster 0
- Stage 1予約: 1 variant / 773 HMM runs / control再実行0

## 再現性設計

- seed policy: real scoreはRNGなし、shuffle controlはwell/window content SHA由来
- stochastic処理: deterministic negative-control permutationのみ
- PF/Beam/likelihood-PF: なし
- CPU/GPU: Kaggle CPU、internet/GPU/TPU off
- input SHA: exp226 OOF、exp280/281 score/decoder contract、fold/hidden-like assignment
- feature SHA: window manifest、score bank、eligibility、lambda、shuffle、readout
- model SHA: fitted modelなし。Stage 1はdecoder contract SHA
- prediction SHA: Stage 1時のみ
- submission SHA: 非該当
- bootstrap: loose/package/bootstrap config一致を確認

## リスク

- 科学リスク: exp305/343ではtempering・correlation由来のevidence調整が不成立だった。
- 親リスク: exp281はexp263 fixedより1.589088 ft悪く、window factorで大幅改善が必要。
- modeリスク: sparse scoreがwrong residual-offset modeを強化し得る。
- leakage: window/score/lambda bundleをsuffix truth/error結合前にfreezeする。
- runtime: Stage 0は軽量、Stage 1は773 HMM runs。

## 2026-07-25 Stage 0実装判断

- 正規`*_train.ipynb` / `*_inference.ipynb`は未承認のplaceholderとして保持し、
  Jupytext percent形式のcompact self-contained候補を別名で実装する。
- 各windowはunknown suffix上のfull 500 rowsだけを採用し、centerは250行目から
  stride 125で置く。
- exp226の既存式に合わせ、known-prefix GRをType Well GRへaffine calibrationし、
  correlationは`2*atanh(clip(corr,-0.95,0.95))`、MSEとlevelは固定sigmaで
  重み付けする。
- 13 shiftのraw scoreをstate方向へ標準化し、そのsoftmax posterior SDから
  `125/500 * clip(1.1-0.12*sd,0.3,1.0)`を計算する。
- exp280 controlは保存済み512-row aggregateを再計算せず、window centerの
  `suffix_offset // 512`で対応付ける。これはscoreの部分block復元ではなく、
  事前固定されたaggregate controlのlookupである。
- window/profile/eligibility/lambda/shuffle/control mappingを一つのbundleへまとめ、
  content SHA確定後だけexp226 OOFの`tvt_true`を読む。

## 2026-07-25 Stage 0結果

- Kaggle private CPU version 1、773 wells、3,783,989 rowsを完了した。
- score finite、row identity、saved-control rank parity、quantization coverageは各1.0。
- eligible window fractionは`0.385617`、real>shuffleは5/5 foldsで、
  実装・coverage・negative controlは成立した。
- saved exp280 control比はpooled MRR `-0.022264`、top3 `-0.033496`。
  MRR/top3改善foldは各0/5、long-tailとhidden-like 2面もすべて負方向だった。
- 全10,628 eligible windowsでlambdaが固定下限`0.075`へ飽和した。
  ただし同一window内のshift rankは正の定数lambdaで変わらないため、
  主な失敗原因は500-row profile score自体のcontrol比不足と判断した。
- 事前禁止したwindow/stride/weight/lambda救済gridは行わず、
  Stage 1、inference、submissionへ進まない。
