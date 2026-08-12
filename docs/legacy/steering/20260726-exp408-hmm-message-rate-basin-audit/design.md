# 設計

## アプローチ

exp270でparity済みのexp209 forward-backwardを1 wellずつ再生する。forward中に
emission適用前のpredictive jointと適用後のfiltered jointを正規化して、position別
mass/rate一次・二次momentとrate marginalへ縮約する。backward中はbetaを永続化せず、
smoothed jointをalpha bufferへ上書きする。これによりfull alpha/betaを保存せず、
late truth maskで任意のposition basin、rate近傍、covarianceを計算できる。

各wellでは次の順を強制する。

1. `TVT`を列選択から除外したraw horizontal、typewell、保存済みexp270の
   target-free posterior mean / Viterbiだけを読む。
2. current HMM messageを計算し、posterior mean parityを確認する。
3. predictionとmessage sufficient-statisticsのSHAをfreezeする。
4. その後だけraw `TVT`とpersistent episode詳細を読む。
5. truth / mean / Viterbi basinを同じ±5 ftで集計し、well終了時に巨大tensorを破棄する。

## 実験範囲

- 対象実験: `exp408_hmm_message_rate_basin_audit`
- Route: `pf_beam`
- 科学的親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 忠実性参照: `exp270_exact_hmm_posterior_mode_candidate_audit`
- 補助参照: `exp391_prefix_anchored_mode_persistence_hmm_readout`
- 変更する変数: なし。内部messageの観測とstream集計だけを追加する。
- 固定する変数: exp209 HMM scientific contract全体、対象450 wells、basin半径5 ft、
  rate近傍±1 state、episode manifest、分類規則。

## 観測量

- position: predictive / filtered / smoothed の truth / mean / Viterbi basin mass、
  mean、variance、edge mass
- rate: 各段階の mean、variance、edge mass、truth / mean / Viterbi path-rate近傍mass
- interaction: basin内 conditional rate mean/variance、position-rate covariance
- attribution: emission前後とfilter/smooth間のtruth-vs-mean、truth-vs-Viterbi log-odds差
- transition: current kernel expected displacement/variance/quantization bias、
  同じactual filtered source-rate massでのcurrent-minus-exact-mean期待変位
- multiplicity: predictive/filtered forward logsum-minus-max gap
- timing: truth-basin escape、filter rescue、beta reversal、smoothed re-capture

## 原因分類

優先順位を固定し、episodeを1つだけへ分類する。

1. `state_support_shortage`: truth position/rateのsupport外率が10%以上
2. `backward_smoothing_reversal`: filteredではtruth優位だがsmoothedでmean側へ反転
3. `raw_gr_alias`: observed GR emissionがtruth-vs-mean oddsを継続的に悪化
4. `imputation_alias`: missing GR行のemission相当変化が支配
5. `forward_transition_prior_hysteresis`: emission前からmean側が優位で、
   emission/beta反転では説明できない
6. `sum_product_path_multiplicity`: Viterbiがmeanより明確に改善し、
   logsum-max gapとmean/Viterbi乖離が大きい
7. `mixed_or_unresolved`

分類閾値はconfigに固定し、結果を見て変更しない。

## 再現性設計

- seed policy: 乱数なし。well / rowは辞書順、Numba reduction順は固定する。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging: なし。
- 並列処理: outer worker 1、Numba thread 4。RNGとの関係なし。
- runtime: Kaggle CPU、GPU/internet無効、9時間hard limit、25 GB RSS guard。
- input SHA: target-well/episode manifest raw SHA、exp270 gzip decompressed SHA、
  target-free raw identity SHAを記録する。
- prediction/message SHA: wellごとのfreeze SHAと全体logical SHAを記録する。
- model/submission SHA: modelとsubmissionを作らないため対象外。
- bootstrap: package直下とbootstrap内configのbyte一致をpush前に検証する。

## リスク

- リークリスク: scopeの450 well IDだけは事前固定入力として許可する。episode境界とtruthは
  well message freeze後にのみ読む。decoder関数はtruth引数を持たない。
- CV/LB不一致: prediction候補・CV昇格・submissionを作らない原因診断なので対象外。
- runtime: 450 wellsのbase見積り1.876時間。message縮約とCSV保存を含む上限3時間を期待し、
  Kaggle hard guardは9時間。
- memory: well単位でjoint posteriorを保持し、終了ごとに破棄する。25 GBを超えたら停止。
- 数値: posterior mean parity `1e-5 ft`、normalization `1e-5`をfail-closed gateにする。

