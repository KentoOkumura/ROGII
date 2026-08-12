# 設計

## アプローチ

exp209 の状態 `(p,r)` を `(p,r,q)` に拡張する。`q` は `normal / weak` の2状態で、
遷移行列を `[[511/512,1/512],[1/128,127/128]]`、初期確率を`[0.8,0.2]`に固定する。
normal は exp209 log emission をそのまま使い、weak は log emission を0.25倍する。
位置・rate遷移や観測sigmaは変更しない。

Stage 0 は HMM を回さない。保存済みexp209 path上のGR residualから、512行・stride 256の
blockごとに固定2状態forward recursionでweak posteriorを計算する。posteriorとblock ledgerの
content SHAを凍結してから suffix truthをjoinし、exp209 block RMSE 10 ft以上をbad labelとして
AUC、weak-posterior四分位差を読む。within-well circular shiftをnegative controlとする。

実装時の低レベル契約は次で固定する。

- block start は suffix offset `0, 256, 512, ...` とし、512行未満の末尾blockも保持する。
- 各blockは初期確率`[0.8,0.2]`から独立にforward filterを開始し、最初のrowでは
  transitionを掛けずemission update、2行目以降はtransition後にemission updateする。
- primary scoreはblock内forward-filtered weak posteriorの行平均。全体weak massは
  `sum(weak posterior) / sum(block rows)`の行加重平均とする。
- circular controlは`sha256("exp363|<well>")`から得たwell内非ゼロblock shift。
  multi-block wellではscore multisetを保存し、single-block wellだけidentityを許す。
- Q1/Q4境界はtruth join前のpooled weak scoreの25%/75%点で凍結し、
  `Q1 <= q25`、`Q4 >= q75`のmean block RMSE差を読む。境界が同値ならtechnical FAIL。
- 4/5 foldsはreal bad10 AUCが各foldで`>0.50`となるfold数。hidden-likeはspatialと
  typewell-purgedの両方でAUC`>=0.55`を要求する。

Stage 1 はStage 0の全gate PASSと別承認時だけ実施する。1 variantで全773 wellsのexact HMMを
実行し、保存済みexp209 controlと比較する。いずれかのsafety gateを破ったら、係数・遷移・blendで
救済せず閉じる。

## 実験範囲

- 対象実験: `exp363_sticky_gr_reliability_exact_hmm`
- Route: `pf_beam`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 変更する変数: sticky GR reliability stateのみ。
- 固定する変数: position/rate grid、transition、momentum、初期分布、Gaussian sigma、posterior mean。
- Stage 0 gate: AUC `>=0.60`、circular差`>=0.02`、Q4-Q1 RMSE`>=0.50 ft`、
  4/5 folds、hidden-like AUC`>=0.55`、平均weak mass`[0.02,0.50]`。
- Stage 1 gate: exp209比`>=0.05 ft`改善、4/5 folds、1000+/hidden-like/p95回帰
  `<=0.02 ft`、worst回帰`<=0.25 ft`。

## 再現性設計

- seed policy: RNGなし。well、row、stateをstable sortする。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging: なし。deterministic exact HMM。
- 並列処理: single worker。
- CPU/GPU: CPU、GPU off、上限30,600秒 / 25GB。
- SHA: input、block ledger、weak posterior、predictionのcontent SHA。gzipはdecompressed SHA。
- prediction freeze後にtruth/control/hidden-like roleをjoinする。
- Kaggle package: 実装承認後もpush前にconfigとbootstrap内定数を照合する。

## リスク

- リークリスク: bad-block labelをposterior生成へ混ぜる危険。truth-late-joinで防ぐ。
- CV/LB 不一致リスク: validationの不良観測頻度がtestと異なる可能性。
- ランタイム/メモリリスク: qで状態数が2倍。上限超過なら実行を中止する。
- 再現性リスク: 低いが、state orderとlogsumexp順を固定する。
- 科学リスク: robust likelihood系の既存負結果と同様にtailだけ悪化し得る。

## 実行結果

Kaggle private CPU version 1（id_no `128370770`）を完了した。
technical gateはPASSしたが、hidden-like spatial AUC
`0.546057972 < 0.55`とrow-weighted weak mass
`0.589440997 > 0.50`が固定gateをFAILした。decisionは
`stage_0_failed_close_without_rescue`、Stage 1 eligibilityはfalseとし、
transition、multiplier、sigma、block、threshold、blendで救済せずbranchを閉じる。
