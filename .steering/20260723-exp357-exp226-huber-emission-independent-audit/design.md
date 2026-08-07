# 設計

## アプローチ

exp226 `tvt_geop`を中心とする固定13-shift bank上で、exp281 sigmaを使った
standardized residualへ固定Huber lossを適用する。Gaussian exp280 scoreを保存controlとし、
Student-t結果を見てdeltaを選ばず、旧exp344で事前登録済みの`1.345`だけを使う。

Stage 0はtarget-free score/rank監査で、prediction/HMMを生成しない。exp281自体が
exp263 fixedに敗れているため、Stage 0 PASSだけでは科学採用せず、Stage 1でも
exp226単体以下というabsolute ceilingを追加する。

2026-07-24のStage 0実装後、ユーザーの明示overrideによりStage 1を同じ
compact self-contained trainへ追加する。exp281のoffset/rate state、transition、
prior、sigma、missing policy、posterior meanを固定し、行別Gaussian emissionだけを
fixed Huber `delta=1.345`へ置換する。inference候補は明示停止を維持する。

## 実験範囲

- 対象実験: `exp357_exp226_huber_emission_independent_audit`
- Route: `pf_beam`
- 親実験: `exp281_exp226_residual_offset_exact_hmm_transition_probe`
- 保存control: exp280 Gaussian shift score
- 変更する変数: Gaussian lossからfixed Huber lossへの置換だけ
- 固定する変数: exp226 shape、512-row block、13 shifts、fold、missing、
  exp281 sigma、state grammar、transition、prior、output
- Huber式:
  - `0.5*z^2` if `|z|<=1.345`
  - `1.345*|z|-0.5*1.345^2` otherwise
- Stage 0: scientific score 1 / saved control 1 / 5 folds / HMM・model・booster 0
- Stage 1 override: 1 variant / 773 HMM runs / parent/control再実行0

## 再現性設計

- seed policy: RNGなし、well/block/shift/fold順を固定
- negative control: content SHAから固定したnonzero circular shift
- stochastic処理: なし
- CPU/GPU: Kaggle CPU、internet/GPU/TPU off
- input SHA: exp226 OOF、exp280 Gaussian score、exp281 sigma/contract
- feature content SHA: Huber score、block readout、control parity、gate
- model SHA: fitted modelなし。Stage 1 decoder contract SHAを記録
- prediction SHA: Stage 1 OOFのraw gzip / decompressed / logical content SHAを記録
- submission SHA: 非該当
- bootstrap: loose/package/bootstrap config一致をpush前に確認

## リスク

- 科学リスク: exp342のrobust emissionはpooled gainが小さくstressで悪化した。
- 親リスク: exp281 direct RMSE 9.827420でexp263 fixedより1.589088 ft悪い。
- tailリスク: robust lossでwrong modeの罰則が弱まり長いoffsetを維持し得る。
- leakage: score bundleとcontrolをtruth/error結合前にfreezeする。
- runtime: Stage 1はKaggle CPUで773 HMM runs、exp342実績から約4時間を想定する。

## 実行結果

Kaggle CPU version 2でfixed Huber exact-HMMを773 wellsに実行した。
Huberはsaved exp281 GaussianをRMSE `9.827420 -> 9.737195`へ
`0.090225 ft`改善し、4/5 foldsと1000+・hidden-like 2面を改善した。
一方、by-well p95は`+0.003365 ft`、worst wellは`+1.403715 ft`悪化し、
exp226 direct ceilingにも`+0.310086 ft`届かなかった。
設計時に固定したAND gateどおり、救済・inference・submissionなしで閉じる。
