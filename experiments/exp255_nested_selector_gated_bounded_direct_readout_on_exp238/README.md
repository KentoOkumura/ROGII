# exp255 nested selector gated bounded direct readout on exp238

## 状態

Kaggle CPU train v1は完了しました。固定3 profileはいずれも採用guard不通過で、推論は禁止のままです。

- kernel: `kentookumura/exp255-gated-bounded-selector-readout-train`
- version: 1
- accelerator / internet: CPU / off
- training: model config 0 / fold 0 / booster 0
- submission: 生成・提出ともになし

## 仮説

exp238のadd-only最終予測をbaseに、fold-held-out selectorが1位とした候補へ明示的に補正するtrain-side OOF監査です。

無条件hard top1はexp245でworst-wellを+38.016697悪化させたため使いません。selectorの予測誤差gain/margin、target-freeなwell内consistencyでgateし、補正量を固定上限へclipします。

## 検証方針

- active audit: 1
- fixed profiles: 3
- model config / fold training / booster: 0 / 0 / 0
- parent/control再学習: なし
- runtime: Kaggle CPU、internet off
- inference / submission: OOF guard通過まで禁止

outer foldのrole=`valid` selector scoreだけで候補を選択します。truthは3 profileの予測を固定した後にmetric計算へ使います。global、near、longtail、hidden-like、fold、worst-wellのguardを全通過したprofileだけを後続推論候補にします。

## 所見

assertiveはglobal RMSEを`7.936690 -> 7.877990`（`-0.058700`）改善し、near、1000+、hidden-like 2面、3/5 foldsでも改善しました。一方、worst-wellは`+3.151245`、`+0.25 ft`超悪化は106 wellsで、唯一の未通過guardがwell-tail riskでした。

無条件hard top1は`8.512262`（base比`+0.575572`）なので不採用です。target-freeなwell consistencyだけではselector誤選択のtail riskを除けませんでした。

## 次アクション

exp255のinference / submitは行いません。再訪する場合だけ、outer-train wellsで学習したwell-risk discriminatorを別実験として監査し、worst-well `+0.25 ft` guardを維持します。
