# 設計

## アプローチ

exp209 exact HMMを1回そのまま実行し、truthを読む前にfiltered / smoothed messageから
persistent rate-disagreement eventをfreezeする。2回目のexact HMMではrate transitionを
一切変えず、最初のeventへ入るposition transitionだけを3枝へ分岐させる。

3枝の状態を`datum_branch ∈ {negative, parent, positive}`として明示的に保持する。
event前はparent枝だけが存在し、event transitionで次のprior massを与える。

```text
negative reanchor: 0.10
parent datum:      0.80
positive reanchor: 0.10
```

reanchor幅はfirst-pass event rowのfiltered position標準偏差を使い、

```text
datum_shift_ft = max(filtered_position_std_ft, position_grid_step_ft)
position_grid_step_ft = 0.35
```

と固定する。negative / positive枝はevent transitionのposition kernel中心を
それぞれ`-datum_shift_ft` / `+datum_shift_ft`だけ移動する。rate stateは同じ
source / destination rateを共有し、event後は3枝とも親と同じrate / position
transition、GR emissionで独立に進む。枝間の再遷移、2回目以降のreanchor event、
hard branch selectionは行わない。

forward-backwardはbranch dimensionを含む同じexact sum-product recursionで実行する。
最終TVTは3枝を周辺化したposition posterior meanとする。branch posterior massは
診断用に保存し、truth join後だけsoft-selected datum方向の正しさを評価する。

## 対称枝を選ぶ根拠

exp412 Stage 0ではfuture betaのrate修正方向が`0.776347`、4 / 5 foldsで正しかったが、
rate transitionを固定10%変更したbackward-cause SSEは`6.96%`悪化した。

設計前診断としてexp408の保存row ledger 807,710行に対し、同一ledger内で
`abs((smoothed_rate-filtered_rate)/max(filtered_rate_std, 0.005)) >= 2`となる
39,873行だけを読み、rate差符号を`truth - posterior_mean`のdatum修正符号へ
直接転用した。一致率は`0.365887`、SSE加重一致率は`0.396557`だった。
1 filtered-position標準偏差を一方向へhard適用したRMSEも
`28.063421 -> 29.441317`と悪化した。一方、parentと同じ1σ枝をtruthでoracle選択した
診断上限は`26.815358`であり、候補余地自体は残った。

この診断はerror-selected episode rows上の設計補助であり、CVや成功根拠ではない。
rate方向をdatum方向へ写像せず、正負対称枝をtarget-free future likelihoodでsoft選択する
設計だけを支持する。

## 実験範囲

- 対象実験: `exp425_symmetric_datum_reanchor_exact_hmm`
- Route: `pf_beam`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 機構証拠:
  - `exp408_hmm_message_rate_basin_audit`
  - `exp412_beta_filter_rate_disagreement_two_pass_reset`
- 変更する変数:
  - first persistent beta-filter rate-disagreement eventでのexplicit datum branch追加
  - branch prior `0.10 / 0.80 / 0.10`
  - branch shift `±max(filtered position std, 0.35 ft)`
- 固定する変数:
  - exp412のfirst-pass disagreement statistic、threshold、window、persistence contract
  - exp209の41-state rate grid、rate transition、momentum、position kernel noise
  - GR emission、position support、initial position/rate prior
  - exact sum-product forward-backward、posterior-mean readout
  - fixed32 manifestとtruth-late join

## Trigger契約

- statistic:
  `z_beta = (smoothed_rate_mean - filtered_rate_mean) /
  max(filtered_rate_std, 0.005)`
- threshold: `abs(z_beta) >= 2.0`
- rolling window: 16 rows
- qualifying rows: 8 rows以上
- same-sign fraction: 0.75以上
- event: persistent activeが初めてfalseからtrueになるrow
- events per well: 最大1
- sign usage: activationの一貫性確認だけに使い、datum branch方向選択には使わない
- freeze: first-pass終了後、truth / episode / errorを読む前

## Stage 0

exp412のSHA固定fixed32を再利用する。

- backward-cause wells: 8
- forward-cause wells: 8
- matched controls: 16
- baseline first-pass exact HMM: 32 well-runs
- treatment branch exact HMM: 32 logical well-runs
- treatment branch state count: 3
- active scientific variants: 1
- LightGBM config / fold / booster / model: `0 / 0 / 0 / 0`
- PF / Beam / GPU: `0 / 0 / 0`

technical gate:

- fixed32 / saved exp209 / first-pass parityを全件確認する。
- first-pass message、event schedule、shift、branch posterior、predictionをtruth前にfreezeする。
- finite coverage `1.0`、normalization error `<=1e-5`。
- event wells `>=8`、全5 foldsにevent wellが存在する。
- peak RSS `<=25 GB`。
- Stage 0実測からのfull runtime projection `<=30,600秒`。

mechanism gate:

- active rowで`positive mass - negative mass`の符号と
  `truth - parent prediction`の符号一致 `>=0.60`。
- strict `>0.50`のfoldが4 / 5以上。
- backward-cause episode SSE reduction `>=0.10`。
- forward-cause episode SSE regression `<=0.02`。
- matched-control RMSE delta `<=+0.02 ft`。
- matched-control branch posterior mass mean `<=0.10`。
- active eventの平均reanchor posterior mass `>=0.05`。

すべてANDで判定する。fixed32はmechanism sampleなので、PASSしてもCV、
promotion、raw-test一般化を主張しない。

## Stage 1

Stage 0のtechnical / mechanism / runtime gateがすべてPASSし、ユーザーが
Stage 1の実装・実行を別途承認した場合だけ設計を再確認する。

- first-pass exact HMM: 773 well-runs
- treatment exact branch HMM: 773 logical well-runs
- branch states: 3
- reporting folds: 5
- baseline comparison: saved exp209 predictionとfirst-pass parity

最低promotion gateはdirect RMSE gain `>=0.05 ft`、改善4 / 5 folds、
backward-cause SSE reduction `>=0.10`、固定hidden-like / 1000+ / GR scopes nonworse、
by-well delta p95 `<=+0.25 ft`、worst well delta `<=+5.0 ft`とする。

exp412 two-passの既存full projectionは`51,753.199秒`で、現行上限を既に超える。
explicit 3-branch treatmentはさらに重いため、本設計はまず機構検証用であり、
deploy可能性を前提にしない。科学gateだけPASSしruntime gateがFAILした場合は、
同一実験内で近似readoutへ差し替えず、exact semanticsを保つ高速化案を別設計・
別承認で扱う。

## 再現性設計

- seed policy: RNGなし。well / row / position / rate / branch順を固定する。
- stochastic処理: なし。
- PF/Beam / likelihood-PF / seed bagging: なし。route分類はexact HMMのため
  `pf_beam`とする。
- 並列処理: outer well順とbranch index順を固定し、Numba parallel reductionの
  加算順を親parity testで監査する。
- runtime: Kaggle private CPU、GPU無効、internet無効。
- SHA:
  - fixed32 manifest
  - saved parent decompressed content
  - first-pass messages
  - activation event / shift schedule
  - baseline / treatment predictions
  - branch posterior masses
  - truth-late metrics
- model / submission SHA: modelとsubmissionを作らないため対象外。
- bootstrap: push前にloose config、embedded config、Notebook body、
  dependency file SHAを照合する。

## リスク

- リークリスク:
  event、shift、branch prior、predictionをfreezeするまでtruth / episode / errorを
  読まない。fixed32選択済みsampleはmechanism診断にだけ使う。
- CV/LB不一致:
  Stage 0はCVではない。full OOFとhidden-like guardなしにinferenceへ進めない。
- ランタイム/メモリ:
  first pass + explicit 3-branch second passで、exp412より高コストになる。
  30,600秒 / 25 GBを緩和しない。
- sum-product multiplicity:
  branch追加がwrong massを増幅する可能性がある。branch priorを合計1へ正規化し、
  eventを1 well 1回に限定し、control branch mass / RMSEをgate化する。
- 再現性:
  float32 message、reduction順、branch indexing差でparityが崩れる可能性がある。
  toleranceを結果後に緩めない。

## 禁止事項

- one-sided datum branch
- exp412 rate directionのdatum sign転用
- trigger / window / persistence / branch prior / shift scaleのgrid
- 2回目以降のreanchor event
- truth / error / cause / well IDによるbranch選択
- hard MAP / Viterbi branch selection
- rate transition、momentum、sig_r、GR emission、support、readoutの同時変更
- branch posteriorのpost-hoc clip / shrink / blend
- same-OOF rescue
- Kaggle実行、Stage 1、inference、submissionの無断着手
