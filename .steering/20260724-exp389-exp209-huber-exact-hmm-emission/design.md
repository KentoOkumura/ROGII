# 設計

## 仮説

exp209 absolute-TVT exact HMMのGaussian row emissionは、大きなGR残差により
wrong modeへ過剰に引かれる場合がある。状態空間と運動・観測scaleを変えず、
emissionだけをfixed Huber `delta=1.345`へ置換すれば、Gaussian direct pathを
well-level tailを壊さず改善できるかを検証する。

## アプローチ

exp209のabsolute-TVT exact forward-backward HMMを科学的親とする。
既存のsaved exp209 Gaussian OOFを再実行しないcontrolに固定し、candidate側だけ
行別Gaussian emissionをHuber `delta=1.345`へ置換する。

exp357で誤って使ったexp281 residual-offset HMM、exp226 `tvt_geop`中心grid、
13-shift rank auditは使わない。実装時の構成参照には同じexp209親を持つ
exp374を使えるが、Student-t式、実測値、tail結果からdeltaやgateを調整しない。

0-HMM proxyはfull HMMの方向を保証しないため設けない。実装・実行が別承認された場合は、
固定Huber 1件を773 wellsで直接decodeし、candidate predictionとlogical SHAを
unknown-suffix truth結合前にfreezeする。その後だけsaved controls、fold、scope、
by-well readoutを結合して固定AND gateを評価する。

## 実験範囲

- 対象実験: `exp389_exp209_huber_exact_hmm_emission`
- Route: `pf_beam`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- sibling reference: `exp374_exp209_student_t_exact_hmm_emission`
- mis-scoped history: `exp357_exp226_huber_emission_independent_audit`
- 変更する変数:
  row emission familyをcapped Gaussianからfixed Huber `delta=1.345`へ置換する。
- 固定する変数:
  absolute-TVT coordinate、grid step `0.35`、41 rate states、rate span contract、
  band pad `100 ft`、`sig_r=0.002`、`sig_p=0.02`、`lam=1.0`、
  start sigma `0.75`、`r0_sig=0.01`、momentum `0.998`、
  known-prefix zero-fill population std、sigma clip `[10,60]`、
  missing-GR補間、Type Well GR補間、posterior mean。
- Huber式:
  `-0.5*z^2` for `|z|<=1.345`、
  `-(1.345*|z|-0.5*1.345^2)` otherwise。
- 追加clip / temperature / normalization:
  `none / 1.0 / state-independent constant omitted`。
- primary control:
  saved exp209 Gaussian exact-HMM `11.938287234887435`。
- secondary control:
  saved Gaussian HMM + LikPF fixed 50:50 `10.269696146642758`。
- 将来実行契約:
  1 variant / 773 HMM well-runs / reporting 5 folds /
  model・LightGBM・trained fold・booster・PF・Beam・parent rerun各0。
- 現在の範囲:
  design、backlog、scaffold、config固定、compact self-contained Jupytext
  train候補、fail-closed inference候補、専用テストまで。
  canonical Notebook採用、package、runは対象外。

## 検証と判定

- technical:
  source/input/control SHA、3,783,989 rows / 773 wells、ID/order/fold、
  finite coverage、posterior normalization、truth-before-freeze 0。
- primary:
  direct gain `>=0.05 ft`、改善4/5 folds、raw observed gain `>=0.05 ft`。
- missing/tail:
  raw missing、高missing wells、1000+、hidden-like 2面をすべて非悪化。
- by-well:
  delta p95 `<=0`、worst regression `<=+0.25 ft`。
- fixed blend:
  candidate HMMとsaved LikPFの固定50:50がGaussian fixed50:50を非悪化。
- 全項目AND。FAIL時は`huber_exp209_failed_close_without_rescue`。
- PASS時もfail-closed inference実装の検討資格だけを与え、実装・実行には別承認が必要。

## 再現性設計

- seed policy:
  RNGなし。well / row / position-grid / rate-grid / variant順を固定する。
- stochastic 処理の有無:
  なし。
- PF/Beam / likelihood-PF / seed baggingの有無:
  candidate生成にはなし。saved LikPFはsecondary reporting controlにのみ使う。
- 並列処理と乱数の関係:
  global RNGを使わず、outer worker数とNumba thread数を固定する。
- CPU/GPU runtime:
  Kaggle private CPU、GPU/TPU/internet off。1 variant / 773 wells。
  exp374実績を踏まえ`11,520--30,600 sec`を計画枠とする。
- input SHA:
  raw well identity、saved exp209 HMM、saved exp072 LikPF、exp226 reporting-fold、
  exp115 hidden-like assignmentをhard checkする。
- prediction SHA:
  raw gzip、decompressed CSV、logical key/value contentを分けて保存する。
- model manifest / submission SHA:
  fitted modelとsubmissionは対象外。decoder/scientific contract、candidate、
  metrics、gate SHAを記録する。
- Kaggle package bootstrap:
  実装が承認された場合だけ、loose/package/bootstrap config、
  self-contained source、metadata、kernel sources、CPU/internet設定をpush前に照合する。
- deterministic anchor:
  train-side candidate SHAは記録するが、inference未実装なのでdeterministic
  submission anchorとは呼ばない。

## リスク

- リークリスク:
  unknown-suffix truth/error/scopeをcandidate freeze前に読むと評価が汚染される。
  exp226からはidentity/foldだけをallowlistし、prediction列をdecoderへ渡さない。
- 科学リスク:
  Huberは外れGRだけでなくwrong stateへの罰も弱め、mode slipを増やし得る。
- tailリスク:
  exp374は平均`0.217809 ft`改善してもp95`+0.982661 ft`、worst
  `+35.015963 ft`だった。Huberも平均改善とwell-tail悪化が併存し得る。
- mis-scopeリスク:
  exp357の`9.827420 -> 9.737195`はexp281 residual-offset HMMの結果であり、
  本実験のcontrolや期待値に使わない。
- CV/LB 不一致リスク:
  exp209 branchは現行Public-LB anchorではない。train PASSでもsubmissionへ直結しない。
- ランタイム/メモリリスク:
  773-well exact HMMはCPUで数時間かかる。candidate 1件、control rerun 0に限定する。
- 再現性リスク:
  浮動小数並列順やgzip metadata差に備え、decompressed/logical SHAと
  metric tolerance `1e-5`を使う。
- multiple-testingリスク:
  `delta=1.345`以外を試さず、結果後のdelta/scale/clip/temperature/grid/
  blend救済を禁止する。

## 次のアクション

Kaggle private CPU version 1を完了した。technical gate、overall、5/5 folds、
全required scope、fixed 50:50はPASSしたが、by-well p95とworst-well gateを
FAILした。事前登録したno-rescue契約に従いterminal closeし、再実行、
inference、submissionへ進まない。
