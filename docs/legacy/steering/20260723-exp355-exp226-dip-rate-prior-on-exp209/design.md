# 設計

## 仮説

fold-safe exp226 K16 geometryの絶対TVTではなく相対rate変化だけをexp209 exact-HMMの
row-wise prior meanへ入れると、constant rate priorより未知suffixのdriftを減らせる。

## アプローチ

exp209 exact-HMMをtrusted controlとし、target-well known-prefixで得た初期rateへ、
fold-safe exp226 K16 geometryの相対rate変化だけを加える。exp323の科学的着眼点を
維持する一方、exp307 finite-MAD、exp308 missing weight、exp309 adaptive `sig_r`、
exp338の全clip scheduleは一切入力しない。

Stage 0ではHMMを実行せず、outer-valid truthを読む前にK16 segment、geometry rate、
fallback、prior scheduleをfreezeする。その後だけactual rate/pathとの一致をfold別、
1000+、hidden-like 2面、by-well tailで評価する。全gate PASS時もStage 1は別承認とする。

2026-07-23の実装では、K16境界をexp226と同じrow-position `linspace` 16分割、
区間rateをfiniteかつ正の`ΔMD` stepの中央値に固定した。先頭geometry区間がinvalidなら
well全体を、後続区間だけinvalidなら当該区間をparent constant rateへfallbackする。
fold gateはsegment rate-changeとcumulative pathの双方で4/5改善を要求する。

2026-07-23、Stage 0は平均では改善したがworst-well guardだけFAILした。ユーザーの
明示overrideを受け、Stage 1を実行する。Stage 0のrow-wise scheduleとgeometry ledgerの
logical SHAをhard guardし、残差rate座標`q_t`で
`effective_dz_t = dz_t - mu_rate_t * dMD_t`としてexp209 exact-HMMへ入れる。
したがって実rateは`mu_rate_t + q_t`となり、観測、diffusion、grid幅、momentum、
posterior meanはexp209から変更しない。

## 実験範囲

- 対象実験: `exp355_exp226_dip_rate_prior_on_exp209`
- Route: `pf_beam`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 変更する変数: exact-HMMのrow-wise rate-prior mean scheduleだけ
- 固定する変数:
  - exp209 Gaussian GR emission、missing interpolation、sigma、state grid
  - `sig_r=0.002`、`sig_p=0.02`、momentum 0.998、start/rate prior、posterior mean
  - exp226 outer-fold assignmentとgeometry-only K16 field
- Stage 0式:
  - `r_geo = Δ(tvt_geop + Z) / ΔMD`
  - `mu_rate_t = parent_initial_rate + r_geo_t - r_geo_first_segment`
  - geometry invalid時はconstant parent initial rateへfallback
- Stage 0実行量: diagnostic 1 / reporting folds 5 / HMM・model・booster 0
- Stage 1予約: 1 variant / 773 HMM runs / control再実行0
- Stage 1実行: user override済み。1 candidate / 773 HMM well-runs /
  model config 0 / trained fold 0 / booster 0 / control再実行0
- Stage 0生成物: truth-free schedule/geometry ledger/fallback、freeze SHA、
  truth late-join後のsegment/path/fold/stress/hidden-like/by-well readoutとAND gate
- Stage 1 freeze境界: exp226 safe columnsとraw known prefixからscheduleを再構築し、
  Stage 0 schedule/ledger SHA一致後に全773 wellの予測をfreezeする。suffix truth、
  saved exp209 prediction、fixed LikPF 50:50 diagnostic、hidden-like roleはその後に結合する。
- 実行境界: compact self-contained Stage 1 train候補を正規train Notebookへ採用し、
  canonical Kaggle CPU version 2を実行する。inference/submissionはfail-closedを維持する。

## 再現性設計

- seed policy: RNGなし、outer-fold / well / K16 segment / row順を固定
- stochastic 処理の有無: なし
- PF/Beam / likelihood-PF / seed baggingの有無: なし
- 並列処理と乱数の関係: RNGなし。固定thread数とstable reduction順を使う
- CPU/GPU runtime: Stage 0/1ともKaggle CPU、internet/GPU/TPU off
- train cache SHA: exp226 OOF decompressed SHAとexp209 control SHAをhard guardする
- feature content SHA: geometry ledger、segment schedule、fallback、fold readoutを記録する
- model manifest: fitted modelなし。Stage 1時はdecoder scientific contract SHAを保存する
- prediction SHA: Stage 1実行時のみraw/decompressed/logical content SHAを保存する
- submission SHA: inference/submission未承認のため非該当
- Kaggle bootstrap: package時にloose / package / bootstrap内configのbyte一致を確認する

## リスク

- リークリスク: exp226 donor fieldへouter-valid wellや同fold valid wellを混ぜない。
- CV/LB不一致リスク: exp281ではgeometry-centered residual decoderがtailを悪化させた。
- 科学リスク: geometry rate-changeがactual rateを説明せず、fold間で符号が反転し得る。
- ランタイムリスク: Stage 0は軽量だが、Stage 1は773 exact-HMM runsで長時間。
- 再現性リスク: exp226 OOFのrow/fold identityとschedule SHA不一致時はfail-fastする。

## 結果と次

Kaggle CPU version 2のStage 1はtechnical gateをPASSし、direct RMSEを
`11.938287 -> 11.291977`へ改善、5/5 folds改善した。一方、hidden-like spatial /
typewell-purgedは`+0.414943 / +0.371720 ft`悪化し、worst wellも
`+52.743754 ft`だったためscientific gateはFAILした。

平均signalは確認できたが未知wellへの安全性がない。事前のfailure actionどおり、
parameter/blend/selector救済、inference、submissionへ進まず、exp355を閉じる。
