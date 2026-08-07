---
title: exp490 再利用戦略調査
date: 2026-08-04
types:
  - experiment_review
  - model_explanation
  - oof_analysis
  - comparison
experiments:
  - exp357
  - exp413
  - exp490
topics:
  - hmm
  - mean_reversion
  - reuse
  - risk
  - ensemble
status: final
summary: "exp490の平均回帰機構、OOFでの利点とtailリスクを整理し、直接blendではなく証拠・risk特徴としての再利用方針を定めた。"
---

# exp490 再利用戦略調査 2026-08-04

## まず着手候補

1. **K16 区間モデル証拠 readout（低コスト・最優先）**
   - exp357 の「回帰なし」と exp490 の「平均回帰あり」を二つの dynamics expert とみなし、各 K16 区間で raw GR / typewell の blocked Huber NLL 差が、どちらを使うべきか予測できるかだけを truth-late で調べる。
   - 新しい HMM / PF / booster / prediction source は作らない。区間 benefit AUC、5 fold の符号、catastrophic-well capture が弱ければ switching branch を閉じる。
2. **exp413 への小さい add-only 機構・risk 特徴（実用候補）**
   - exp490 予測そのものを hard 採用せず、`exp490-exp357`、符号付き物理候補合意、K16 `rho/span/position`、posterior std、nested benefit/risk score を小さい feature block として exp413 final370 に追加する。
   - exp502 は selector block の置換であり、この add-only 仮説とは異なる。親 exp413 は再学習せず、承認後に新規 variant 15 boosters だけを学習する。
3. **position / rate 平均回帰の factorial fixed32（科学的原因分離）**
   - exp490 は residual offset と residual rate の両方へ同時に `rho` を掛けたため、どちらが pooled gain と tail harm を作ったか未識別である。
   - `position-only` と `rate-only` の二候補を fixed32 だけで比較し、保存 exp357 を control とする。複数 variant と CPU コストは実行前に別承認を取る。
4. **二モード switching HMM / transient-bias state（高リスク研究）**
   - `no-reversion` と `mean-reversion` を K16 区間単位で soft に混ぜるか、residual offset を「保持すべき datum」と「平均回帰させる transient drift」に分解する。
   - 1 の観測可能な model-evidence transfer が通ることを先行条件にする。いきなり state を増やさない。

## 結論

exp490 の価値は、単体予測の CV `8.480155` や Public LB `9.680` そのものよりも、**長い suffix で積分された低周波 offset を幾何中心へ戻す遷移則が、複数 route で大きな pooled gain を再現したこと**にある。

一方、固定強度は「真に必要な persistent offset」と「正しい geometry residual offset」を識別できず、同じ well に長く同方向の誤りを作る。したがって今後の中心課題は half-life 調整ではなく、`revert / keep` を観測可能な証拠で区別すること、または最終 ML が correction を連続特徴として安全に利用できる形へ落とすことである。

現時点では exp490 を最終提出へ直接混ぜる根拠は足りない。最終提出準備は既存 P0 `exp509`、P1 `exp510` を優先し、exp490 派生はそれらを追い越さない。

## 現在地

| route / 役割 | 現在の基準 | CV | Public LB |
| --- | --- | ---: | ---: |
| ML submitted anchor | exp413 | 7.884803 | 7.201 |
| ensemble submitted reference | exp082 | - | 7.601 |
| direct LikPF reference | exp452 / exp404 x1.0 | 10.914522 | 8.797 |
| direct exact HMM reference | exp434 exact HMM | 11.938287 | 9.063 |
| geometry baseline | exp226 | 9.427110 | 9.837 |
| mean-reverting exact HMM | exp490 | 8.480155 | 9.680 |

exp490 は exp226 direct より LB を `0.157 ft`改善したが、direct exact HMM より `0.617 ft`悪く、ML anchor exp413 とも大差がある。単体 route anchor にはしない。

## exp490 が実際に示したもの

### 強い正の証拠

- exp357 exact HMM から full OOF を `9.737195 -> 8.480155`、`1.257040 ft`改善した。
- 5 fold 中 4 fold、MD 1000+、hidden-like spatial、typewell-purged を改善した。
- persistent episode SSE を `41.41%`削減し、回復率も改善した。
- 同じ K16 half-life mean-reversion を residual LikPF へ移した exp500 でも、exp404 を `10.914522 -> 8.813505`、`2.101017 ft`改善し、5/5 fold と全固定 scope で同方向だった。
- exp264 selector bank の 13 番目候補にした exp501 は fixed12 を `8.652532 -> 8.264890`、`0.387642 ft`改善し、5/5 fold と全 scope を改善した。

これは偶然の一予測列というより、**translation gauge を持つ residual state に復元力を入れる発想が route 横断で効く**ことを示す。

### 強い負の証拠

- exp490 は 449 wells を改善した一方 324 wells を悪化させ、by-well p95 `+7.257814 ft`、worst `389ae58f +49.602560 ft`だった。
- gain の `36.92%`は上位 10 wells、harm の `57.85%`は worst 10 wells に集中した。
- correction と真に必要な補正方向の alignment は benefit と Spearman `0.820838`だが、これは truth-aware 診断であり inference では直接使えない。
- exp499 の既存 target-free hard well router は always-exp490 より `0.034155 ft`悪く、catastrophic 51 wells 中 48 wells を残した。
- exp500 も p95 `+6.653601 ft`、worst `+46.154671 ft`で、PF へ移しても固定強度 tail は消えなかった。
- exp501 / exp505 の selector 利用でも p95 は約 `+2.90 ft`残った。tau500 fade は pooled を少し改善しても tail を実質除去できなかった。
- exp502 の downstream selector-block replacement は exp413 比 `0.002659 ft`しか改善せず、hidden-like と tail を悪化させた。
- exp506 の `exp490-exp357` additive correction は exp413 を `7.884803 -> 7.902068`へ悪化させ、正の weight は安定しなかった。

## 今回追加した保存 artifact readout

以下は新しい experiment の採用判定ではなく、exp506 の primary gate freeze 後に保存された anchor / exp490 と raw train truth を使った原因調査である。同じ OOF から weight や閾値を選び直して採用してはいけない。

### 1. exp413 と exp490 の固定 10% 直接混合

式は `0.90 * exp413 + 0.10 * exp490`。

| 指標 | exp413 | 固定10% | 差 |
| --- | ---: | ---: | ---: |
| pooled RMSE | 7.884803 | 7.734534 | -0.150269 |
| MD 0--250 | 1.468922 | 1.441795 | -0.027127 |
| MD 250--1000 | 3.889074 | 3.815546 | -0.073527 |
| MD 1000+ | 8.663017 | 8.497857 | -0.165160 |
| hidden-like spatial | 8.364713 | 8.204303 | -0.160410 |
| hidden-like typewell-purged | 8.307715 | 8.158558 | -0.149157 |

- 5/5 folds 改善。
- 536 wells 改善、237 wells 悪化。
- by-well median `-0.097452 ft`、p95 `+0.549195 ft`、worst `+2.657049 ft`。
- 86 wells が `+0.25 ft`超、11 wells が `+1 ft`超悪化。
- 最大悪化 well は exp490 と同じ `389ae58f`。

平均性能と hidden-like は強いが、tail guard は通らない。

### 2. 固定 weight と tail の交換条件

| exp490 weight | RMSE | exp413 gain | nonworse folds | by-well p95 | worst |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 7.867794 | 0.017009 | 5/5 | +0.051827 | +0.253736 |
| 0.025 | 7.843095 | 0.041708 | 5/5 | +0.132028 | +0.639755 |
| 0.05 | 7.804120 | 0.080683 | 5/5 | +0.265396 | +1.296748 |
| 0.10 | 7.734531 | 0.150272 | 5/5 | +0.548974 | +2.656817 |
| 0.20 | 7.629806 | 0.254997 | 5/5 | +1.114248 | +5.528475 |
| 0.40 | 7.563738 | 0.321065 | 4/5 | +2.371980 | +11.690851 |

global OOF 最適 weight は約 `0.368`だが、tail は weight とともに単調に悪化する。1% micro-blend ですら worst `+0.253736 ft`で既存 `+0.25 ft` guard をわずかに越える。これは post-hoc grid なので、1%を救済採用してはいけない。

### 3. LB 相関リスク

exp413 Public LB `7.201`、exp490 `9.680`だけから blend LB は決まらない。10% blend が exp413 LB を改善するために必要な hidden residual correlation は `< 0.710552`である。

OOF の exp413 / exp490 residual correlation は `0.726676`。hidden も同じ相関だと仮定した推定 LB は約 `7.215`で、exp413 よりわずかに悪い。これは推定であって観測 score ではないが、CV `7.7345`だけを根拠に提出枠へ入れない理由になる。

### 4. 符号付き物理候補合意

exp499 の 32 target-free well features は予測差を主に絶対値で集約していた。そこで、6 primitive physical candidates が `exp490-exp357` と同方向かを保存 candidate OOF から集計した。

- correction magnitude 加重 signed agreement: beneficial-well AUC `0.573985`、benefit Spearman `0.170358`、5/5 fold で正。
- 既存最良単変量 `parent_exp226_abs_mean`との Spearman は `0.313354`で、完全な重複ではない。
- 既存 32 features の fixed outer-fold Logistic AUC は `0.659944`、signed 7 features 追加後は `0.671898`。fold 最低も `0.625965 -> 0.646276`。
- ただし hard apply の RMSE は `8.256841`まで改善しても、parent 比 p95 `+1.138018 ft`、worst `+49.602560 ft`で tail を解決しなかった。
- exp490 対 exp357 の `>+5 ft` catastrophic harm は AUC `0.896964`で比較的検知できたが、固定 threshold の二頭 gate でも severe wells を残した。

符号付き合意は **hard router の完成品ではなく、下流 ML の補助 risk feature** として価値がある。

### 5. exp413 固定10% harm の予測可能性

同じ 32+7 features を、固定10% blend が exp413 を改善するか、tail を壊すかへ outer-fold cross-fit した。

| label | positives | pooled AUC | fold 最低 |
| --- | ---: | ---: | ---: |
| fixed10 beneficial | 536 | 0.663014 | 0.613837 |
| delta > +0.25 ft | 86 | 0.710741 | 0.604138 |
| delta > +0.5 ft | 44 | 0.736688 | 0.521523 |
| delta > +1 ft | 11 | 0.552255 | 0.333333 |

benefit / moderate harm は多少予測できるが、最重度 tail は fold 間で安定しない。固定 threshold の二頭 gate は RMSE `7.807--7.819`で always-fixed10 `7.734534`より悪く、worst `+2.657049 ft`も残した。

したがって新しい hard well router は推奨しない。soft feature 化または区間 evidence 監査へ進む。

## 既に試した利用法と判定

| 利用法 | 実験 | 結果 | 今後 |
| --- | --- | --- | --- |
| 単体 exact HMM | exp490 | pooled 強、tail / LB 弱 | 単体採用しない |
| target-free physics regime | exp498 | primary coverage 0 wells | 閾値救済しない |
| cross-fitted well router | exp499 | always-exp490 より悪化 | 同じ hard router を再試行しない |
| PF への遷移則移植 | exp500 | 5/5 fold 改善、tail catastrophic | 固定強度を別 PF へ横展開しない |
| selector 13番目候補 | exp501 | pooled `-0.388 ft`、tail fail | hard candidate bank は閉じる |
| downstream selector block | exp502 | exp413 比 `-0.0027 ft`のみ、scope fail | replacement を再試行しない |
| depth fade / prefix policy | exp503 | tau500 小改善、強い policy 不安定 | cutoff / tau grid をしない |
| faded selector candidate | exp505 | pooled小改善、tail不変 | fade selector を閉じる |
| exp413 への novel correction | exp506 | primary悪化 | `exp490-exp357` scalar加算を閉じる |
| exp413 / exp490 直接 blend | 今回 readout | pooled強、tail / LB risk | nonselectable。提出しない |

## アルゴリズムとして再利用する設計

### A. K16 区間 model-evidence switching

exp490 の平均回帰は well 全体で固定せず、K16 区間ごとに `keep` と `revert` の二 dynamics を持つ方が自然である。

最初の experiment は prediction を選ばず、次だけを確認する。

1. exp357 / exp490 path 上で同じ Huber GR emission を評価する。
2. exp209 系で確認済みの GR evidence 自己相関を考慮し、24 行程度の block 単位へ縮約する。
3. K16 区間ごとの `NLL_keep - NLL_revert` を truth 前に freeze する。
4. truth-late で区間 MSE benefit、fold AUC、catastrophic capture を読む。

この readout が通れば、Dynamic Model Averaging、IMM、switching state-space model のいずれかを固定一式として実装する。通らなければ「観測尤度で mean-reversion mode を選ぶ」branch を閉じる。

### B. position-only / rate-only factorial

exp490 の式は次の二変更を同時に含む。

- rate center: `0.998 * rho_t * q_(t-1)`
- offset center: `rho_t * delta_(t-1) + q_t * dMD_t`

一方、過去の HMM 監査では rate lag が persistent error の形成要因であり、rate をゼロ方向へ弱めることは真の moving rate を抑える危険がある。exp424 の momentum=1、exp441 の full-support OU、exp411/412 の innovation / beta de-stick は単独では persistent 修復に失敗した。

それでも exp490 は position と rate を同時変更したため、`position-only` が gain の中心か、`rate-only` が safety / harm へ寄与したかは未識別である。二候補 fixed32 なら、half-life grid をせず原因を分離できる。

### C. transient drift と persistent datum の二成分化

固定 OU は全 residual offset をゼロへ戻すため、正しい geometry mismatch も消す。概念上は

`residual offset = persistent datum b_t + transient accumulated drift e_t`

と分け、`b_t`は K16 単位の slow random walk、`e_t`だけを mean-revert させるべきである。

ただし GR emission が `b+e`しか観測しないため識別性が弱く、exact HMM は state explosion を起こす。A の model-evidence が正なら switching / Rao-Blackwellized PF として検討し、そうでなければ実装しない。

### D. downstream ML の drift / risk feature

exp264 系の成功例では、物理 candidate は hard path より「危険度・regime の連続特徴」として効いている。exp490 でも次の小 block が候補になる。

- `d490 = exp490 - exp357`、`abs(d490)`、一次差分、区間内 cumulative / monotonicity。
- `exp490-exp226`、`exp490-exp413`、primitive candidates との signed agreement count。
- `rho`、K16 span / position、posterior std。
- strict nested beneficial probability、moderate-harm probability。
- `risk * d490`などの bounded interaction。ただし hard threshold prediction は作らない。

exp502 の compact77 replacement ではなく、final370 を固定した add-only にする。15 boosters を使うため、現行 final-slot 作業後にユーザー承認を取る。

### E. 学習時の teacher / augmentation

exp490 correction を pseudo-target、hard-case weight、candidate perturbation に使う案はある。しかし exp248 candidate perturbation、exp239/244 pseudo-tail augmentation、exp258 GR residual augmentationは総じて tail / hidden-like を安定改善しなかった。

再訪するなら data augmentation ではなく、D の nested auxiliary target に限定する。exp490 の truth-aware benefit を validation feature へ漏らさず、outer-train 内だけで risk head を作る。

## 優先しない案

- half-life、fade tau、blend weight、threshold の同一 OOF grid。
- well ID、worst-well ID、truth / early suffix truth を使う gate。
- exp501 と同型の hard 13th-candidate selector。
- exp500 と同型の固定 mean-reversion を別 PF / Beam へ追加。
- posterior std 単独の uncertainty gate。benefit との関係が弱い。
- suffix 1024+だけの hard switch。fold 0 は長距離でも悪化し、well tailを識別しない。
- 1% micro-blend の救済採用。post-hoc で、tail guardもわずかに失敗する。
- exp490 単体または10% blendの最終提出。Public LB 乖離と相関リスクが大きい。

## 外部研究との対応

- Switching state-space model は、HMM の離散 regime と線形 dynamics を組み合わせ、時系列を異なる dynamics へ分割する枠組みを与える。exp490 では `keep / revert` regime に対応する。  
  <https://www.cs.toronto.edu/~hinton/absps/switch.html>
- IMM は Markov switching system で増える仮説を近似的に merge し、比較的低い計算量で複数 dynamics を扱う。  
  <https://doi.org/10.1109/9.1299>
- Dynamic Model Averaging は model probability を時変にし、posterior predictive probability で model-specific prediction を混ぜる。K16 segment evidence を通した後の soft mixing 候補である。  
  <https://sites.stat.washington.edu/raftery/Research/PDF/Karny2010.pdf>
- Adaptive Mixtures of Local Experts は expert と gating network を共同利用する原型。ただしローカル結果では hard gate の AUC 改善だけでは tail 安全性にならない。  
  <https://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf>
- Bayesian Online Changepoint Detection は直近 change point からの run length posterior を持つ。exp490 の persistent drift onset を区間 evidence として表現する場合の参照になる。  
  <https://arxiv.org/abs/0710.3742>
- Nonnegative cross-validated stacking は異なる predictor を線形結合する標準的根拠を与えるが、今回の固定 blend は well-tail と CV/LB shift を別途満たさなかった。  
  <https://statistics.berkeley.edu/sites/default/files/tech-reports/367.pdf>

## 事実と仮説の境界

### 事実

- exp490 の pooled / persistent 改善と by-well tail 悪化。
- exp500 の route 横断再現と同じ tail 悪化。
- exp498/499/501/502/503/505/506 の negative / bounded results。
- 今回の固定 blend、weight curve、signed-consensus AUC、hard-gate readout。
- exp413 / exp490 の公開 LB と OOF residual correlation。

### 仮説

- K16 blocked Huber NLL が `keep / revert`選択へ転移すること。
- position-only が exp490 gain の主成分であること。
- exp490 mechanism/risk features が exp413 add-only で安定改善すること。
- transient/persistent 二状態が fixed OU tail を抑えること。
- hidden residual correlation が OOF と同程度であること。LB `7.215`はこの仮定下の推定で、実測ではない。

## 推奨順序

1. exp509 / exp510 の既存 final-slot 作業を維持する。
2. 0-model の K16 segment model-evidence readoutを別 steering / 別承認で行う。
3. readout が弱ければ switching HMM を作らず、exp490 は add-only risk feature候補だけに限定する。
4. 実用改善を狙う場合は exp413 control を再学習せず、新規 add-only 15 boosters だけを承認後に実行する。
5. position/rate factorial は科学的価値は高いが、最終提出より後、または明確な研究優先指示がある場合に行う。
