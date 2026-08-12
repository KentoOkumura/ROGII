# 設計

## アプローチ

exp209状態を`(p,r,c)`へ拡張し、`c=-1/0/+1`をpersistent curvature signとする。
exp209 rate grid 1 cellは`0.005`なので、`c`による期待rate driftを1行あたり
`c * 0.005 / 512`に固定する。`c=0`は確率`1023/1024`で維持し左右へ各`1/2048`、
`c=±1`は`511/512`で維持し`1/512`で0へ戻る。直接の符号反転は許さない。

Stage 0ではvisible prefix末端のexp209 rateでanchorした`-1/0/+1`の512-row固定軌道を
stride 256で作り、typewell GR emissionだけで順位付けする。pathとscoreをfreeze後、true TVTに
最も近い符号をlabelとしてtop1/MRRを評価する。geometryから符号やrateを予測しない。
within-well circular-shift GRをnegative controlとする。16 wellsでexact state数3倍のresource
microbenchmarkも行う。

Stage 1は全gateと別承認時だけ773 wellsのexact HMMを実行する。結果が安全gateを破れば
magnitude、transition、sigma、blendを調整せず閉じる。

## 実験範囲

- 対象実験: `exp364_signed_curvature_exact_hmm`
- Route: `pf_beam`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 変更する変数: persistent signed-curvature stateだけ。
- 固定する変数: exp209 position/rate grid、noise、momentum、emission、sigma、出力。
- Stage 0 gate: top1`>=0.40`、MRR gain`>=0.01`、circular差`>=0.03`、
  4/5 folds、1000+とhidden-like 2面が正方向、runtime`<=30600 sec`、RSS`<=25GB`。
- Stage 1 gate: exp209比`>=0.05 ft`、4/5 folds、1000+/hidden-like/p95回帰
  `<=0.02 ft`、worst回帰`<=0.25 ft`。

## 再現性設計

- seed policy: RNGなし。well / row / cの順序を固定。
- stochastic 処理: なし。
- PF/Beam / seed bagging: なし。deterministic exact HMM。
- 並列: single worker。
- CPU/GPU: CPU、GPU off。Stage 0 resource gateを超えたらStage 1禁止。
- SHA: path bank、GR score、state manifest、predictionのcontent SHA。gzipはdecompressed SHA。
- truthはpath/score/prediction freeze後にjoinする。
- bootstrap: push承認後にconfigのstate順、drift、transitionを照合する。

## リスク

- リークリスク: true curvature signで軌道を選ぶ危険。候補とscoreの事前freezeで防ぐ。
- CV/LB不一致: train suffixの曲率頻度がtestと異なる可能性。
- runtime/memory: 状態数約3倍。hard resource gateを置く。
- 再現性: logsumexp順序差。state orderとsingle workerで固定する。
- 科学リスク: richer dynamicsの既存実験は1000+ / worst-wellを壊している。
