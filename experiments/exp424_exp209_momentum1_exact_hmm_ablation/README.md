# exp424_exp209_momentum1_exact_hmm_ablation

## 状態

- ルート: PF/Beam
- 状態: Stage 0 technical PASS / mechanism FAIL・branch閉鎖
- CV / Public LB / Private LB: なし
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 原因証拠: `exp408_hmm_message_rate_basin_audit`
- 作成日: 2026-07-28

## 仮説

exp209 exact HMMではrate posteriorの絶対値が真値より小さい0方向under-responseが
persistent offsetの主要経路に含まれる。`sig_r=0.002`を維持したまま
`mom=0.998`だけを`1.0`へ変えれば、rate diffusionを増やさずに0方向収縮だけを除き、
rate誤差の積分によるTVT offsetを減らせる可能性がある。

## 単一変更

- 変更: `mom: 0.998 -> 1.0`
- 固定: `sig_r=0.002`、rate / position grid、GR emission、prior、
  forward-backward、posterior mean
- 混ぜない: exp338のrate noise変更、exp411 / exp412のtrigger付きtransition変更

## 検証方針

Stage 0はexp411のSHA固定済み32 wells
（persistent 16 / matched control 16）でmechanismだけを確認する。
sample-matchedなrate momentは保存されていないため、
baseline 32 + treatment 32 = 64 HMM runsを比較する。
baseline TVTは保存済みexp209 predictionとのparityを確認する。
このsampleはerror情報を使って選ばれているためCVやpromotion根拠にはしない。

Stage 0全gate PASSと別承認後だけ、773 wellsのfull OOF treatmentを実行し、
exp209 direct HMM、persistent episode、5 folds、1000+、hidden-like、
by-well tail、fixed LikPF/HMM blendをAND評価する。

## Stage 0結果

Kaggle private CPU Version 1（id `128924158`）でbaseline 32 +
treatment 32 HMMを完走した。runtimeは`2,077.533832秒`、peak RSSは
`1.030926 GB`。technical gateは13 / 13 PASSした。

mechanism gateは3 / 7 PASSだった。

- persistent episode SSE削減: `0.475550% < 5%`
- persistent改善well: `8 / 16 < 10 / 16`
- persistent改善fold: `3 / 5 < 4 / 5`
- under-response SSE share低下: `9.849995 points >= 2 points`
- control pooled delta: `-0.054769 ft <= +0.02 ft`
- control by-well p95: `+0.157066 ft <= +0.25 ft`
- smoothed rate edge mass delta: `+0.000377954`でnonworse FAIL

## 所見

rate under-response自体は減ったが、その改善はpersistent TVT offsetへ十分かつ
fold横断に転移しなかった。`mom=1.0`単独branchは`stage0_fail_closed`とする。

## 実装

- compact self-contained train:
  `exp424_exp209_momentum1_exact_hmm_ablation_compact_selfcontained_train.py`
- 正規train Notebook:
  `exp424_exp209_momentum1_exact_hmm_ablation_train.ipynb`
- inference:
  fail-closed placeholderのみ
- contract test:
  `experiments/exp424_exp209_momentum1_exact_hmm_ablation/tests/test_exp424_exp209_momentum1_exact_hmm_ablation.py`

train Notebookはexp209と同じHMM入力準備、3-state rate transition、
position transition、forward-backward、posterior-mean readoutを自己完結で持ち、
baseline / treatmentのpredictionとpredictive / filtered / smoothed rate momentを
32 wellsすべてでfreezeしてからtruthとepisodeをjoinする。

## 実行境界

- 実装・正規Notebook採用: 承認済み・完了
- Kaggle Stage 0（64 HMM runs）: 2026-07-28 完了・FAIL
- Stage 1（773 treatment HMM runs）: 不適格・実行しない
- inference / submission: 無効
- model / booster / PF / Beam / GPU: 0

## 次

同一OOFでmomentum、`sig_r`、gate、sample、blendを救済しない。Stage 1、
inference、submissionへ進まず、完了済みとしてbacklogから削除する。
