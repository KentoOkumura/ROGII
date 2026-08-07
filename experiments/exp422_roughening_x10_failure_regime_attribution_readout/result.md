# exp422_roughening_x10_failure_regime_attribution_readout 結果

## 状態

Kaggle private CPU audit version 2完了。technical PASS / scientific FAILで、
target-free attribution branchを終了した。

## 仮説

roughening x10の局所改善は高いPF回復圧力に、全体悪化は高い欠損・長suffixの
損傷露出に対応し、target-freeな2軸well regimeで符号分岐を再現できる。

## 設定

- 親: `exp416_roughening_x10_likpf_full_oof_ablation`
- control: 保存済みexp072 `likpf_mean`
- validation: exp226 folds 0--4、773 wells / 3,783,989 rows
- primary target cell: high recovery pressure / low damage exposure
- permutation: scoreごとに4096回のfold内置換
- prediction / PF / model / booster / HMM / Beam / GPU: 0

## 変更点

exp416 / exp072 / exp226の保存生成物だけを使い、outcome読込前にfold-safeな2 score、
median cell、row scopeをfreezeした。新しいPF path、予測、モデルは生成していない。

## 結果

| メトリック | 値 | 判定 |
| --- | ---: | --- |
| technical gate | PASS | 入力SHA、truth-late freeze、parity、実行量をすべて確認 |
| scientific gate | FAIL | 全scientific checkをAND評価 |
| exp416 candidate RMSE | 13.617717558 | - |
| exp072 control RMSE | 11.594894396 | - |
| candidate - control | +2.022823162 ft | 5/5 foldsで悪化 |
| recovery-pressure pooled rho | -0.166697697 | 期待した正方向と逆、0/5 positive folds |
| recovery-pressure one-sided p | 1.000000000 | FAIL |
| damage-exposure pooled rho | -0.041484753 | 閾値 -0.10に未達 |
| damage-exposure one-sided p | 0.111300952 | FAIL |
| target cell row-weighted gain | -1.852449584 ft | 1/5 foldsだけ改善、FAIL |
| target cell - rest equal-well gain | -0.518965568 ft | FAIL |
| target cell improved-well fraction | 0.314049587 | FAIL |
| target-cell persistent episode SSE gain | 45.801967% | 支持 |
| positive episode SSE gainのtarget-cell share | 39.400617% | 50%閾値に未達、FAIL |
| CV / Public LB / Private LB | - / - / - | inference・submissionなし |

固定target cellには242 wellsが入り、そのうちpersistent-offsetは4 wells /
4 episodes / 14,827 rowsだった。episode内ではSSEを45.80%減らしたが、全positive
episode gainの39.40%しか説明せず、global target-cell gainとequal-well gainは
ともに負だった。

## 再現性

- Kaggle kernel:
  `kentookumura/exp422-rough-x10-regime-attribution-train` version 2 /
  id_no `128921651`
- status / runtime / peak RSS: `COMPLETE` / 362.877 sec / 3.298 GiB
- scientific contract SHA:
  `20d2644085334ed0028ff8ca0caa38d6379073980f3547c6d05b1f7eee410426`
- artifact manifest SHA:
  `c2fe9339994e8785bf33dc0585f985d5a819e1ff6bf653262bad46e108c04f16`
- feature / assignment / row-scope logical SHA:
  `0fed1d9ed954f6e585f6b8bcdd60c966bc58c67efd6660f7adadb5bbd97dccf4` /
  `e459e3a438511710e48c8646ea1f283f54b1310729cd5e99aee9addd8e7b2fb2` /
  `242562b0d7a05fef042db29ad41b736991548825aebe26b773e759cc29fd194c`
- outcome読込前のtruth / control / by-well / episode rows: すべて0
- model / prediction / submission SHA: 新規生成なし

version 1は親logical SHAの比較列を4列ではなく8列に広げた実装不一致でERRORに
なった。親と同じ4列へ修正し、科学式・入力・gate・実行量を変えずversion 2を完走した。

## 解釈

事前固定した「高い回復圧力かつ低い損傷露出」というwell regimeは、exp416の
局所回復とglobal破壊を再現可能には説明しなかった。特にrecovery-pressureは
期待方向を単に弱く外したのではなく、5 foldsすべてで逆方向だった。

したがってexp416の`roughening_x10_rejected_close_without_rescue`を維持し、
score weight / transform、median、target cell、roughening倍率、position/rate、
ESS thresholdを同一OOF後に調整しない。adaptive rougheningのpolicy実験、
inference、submissionにも進まない。

## 次

exp422内の救済実験は行わず、attribution branchをterminal closeする。別仮説を扱う
場合は、exp416/exp422の結果を固定した独立実験として事前登録する。
