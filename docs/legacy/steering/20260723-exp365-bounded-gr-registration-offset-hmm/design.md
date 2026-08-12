# 設計

## 仮説

物理TVTが正しくてもhorizontal GRとtypewell GRの登録位置だけが小さく持続的にずれるなら、
物理位置`p`を動かさずlookup offset `delta`を周辺化することで、delta=0よりknown-prefix
held-out GRの予測NLLを改善できる。

## アプローチ

exp209状態を`(p,r,delta)`へ拡張する。`p`はphysical TVT position、`delta`はGR/typewell
registrationだけに作用し、emissionを`typewell_GR(p+delta)`で評価する。予測出力は常に`p`。
deltaは`[-6,-3,0,3,6] ft`、初期確率`[0.05,0.15,0.60,0.15,0.05]`。
各行で隣接cellへ移る確率を方向ごとに`1/512`とし、境界の無効遷移massはstayへ戻す。

Stage 0はvisible prefixだけを使う。128行historyでdelta posteriorを作り、続く64行のknown GRを
予測し、stride 64でrollingする。delta=0比のpredictive NLLを主指標とし、held-out GRの
within-well circular shiftをnegative controlにする。未知suffix TVTは読まない。

実装では、history内だけのGRからexp209 `std`式のsigmaを計算し`[10,60]`へclipする。
history missing GRはhistory内補間、held-outはraw finite GRだけをone-step-aheadで
予測・posterior更新する。negative controlはwell内のfinite known-prefix GRを64観測値
circular shiftし、missing maskと値multisetを保つ。foldはstable well順GroupKFold。
16 resource wellsはparent state-cell workload分位から両端を含めて固定し、runtimeは
exp209 v5 `11285.868 sec`をoffset state数5で乗じる。RSSは実際のsuffix / position /
rate shape、5-state alpha/emission/workspace、1GB overhead、1.25 safety factorから投影する。

Stage 1は全gateと別承認時だけ全773 wellsのexact HMMを実行する。deltaは出力に足さない。
失敗時はgrid、transition、sigma、DTW/affine、blendで救済しない。

## 実験範囲

- 対象実験: `exp365_bounded_gr_registration_offset_hmm`
- Route: `pf_beam`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 変更する変数: bounded GR registration offset stateだけ。
- 固定する変数: physical position/rate dynamics、grid、Gaussian sigma、posterior mean。
- Stage 0 gate: NLL gain`>=1%`、4/5 folds、circular差`>=0.5%`、
  nonzero mass`[0.05,0.50]`、boundary mass`<=0.25`、隣接window符号一致`>=0.60`、
  projected runtime`<=30600 sec`、RSS`<=25GB`。
- Stage 1 gate: exp209比`>=0.05 ft`、4/5 folds、1000+/hidden-like/p95回帰
  `<=0.02 ft`、worst回帰`<=0.25 ft`。

## 再現性設計

- seed policy: RNGなし。well / row / delta順を固定。
- stochastic処理、PF、seed bagging: なし。
- CPU single worker、GPU off。
- rolling-window ledgerとdelta posteriorのcontent SHAを保存する。
- Stage 1ではprediction SHA freeze後にsuffix truth/controlをjoinする。
- gzipはdecompressed content SHAを主証拠にする。

## リスク

- リーク: prefix held-out GRをposterior fitへ混ぜる危険。history/held-out境界を固定する。
- CV/LB不一致: testのregistration分布が異なる可能性。
- runtime/memory: state数5倍。resource projectionをhard gateにする。
- 再現性: delta order、transition order、single workerを固定する。
- 科学リスク: deltaがphysical TVT errorを代理して過適合し得る。

## 生成物

- target-free rolling-window ledger
- history / held-out更新後のdelta posterior
- safe input identity manifest
- 16-well exact-state resource projection
- scientific contract / freeze manifest / fold metrics / gate report / summary

## 次のアクション

Kaggle実行は別承認まで停止する。Stage 0を実行して全gateを通過した場合も、
Stage 1実装と773-well exact HMMはさらに別承認へ戻す。
