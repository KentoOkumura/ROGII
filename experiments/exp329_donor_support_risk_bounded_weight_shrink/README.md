# exp329_donor_support_risk_bounded_weight_shrink

## 状態

- Route: `pf_beam`
- 状態: Kaggle CPU Stage 0完了・scientific gate FAIL・branch closed
- 親: `exp263_last_anchor_better_candidate_confidence_pair_cache`
- anchor: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`

## 仮説

exp226の外れは近傍wellの誤差方向をコピーして直すのではなく、fold-safe donorの不足・遠さ・局所平面の不安定性が高いK16 segmentで、exp263固定式に含まれるexp226の寄与を少し減らす方が安全である。

riskは次の6項目をouter-trainだけのpercentileへ変換し、平均後にもう一度percentile化して`[0, 1]`へ固定する。

- donor距離`d15`が大きい
- kernel weightの`n_eff`が小さい
- ridge込みlocal-linear normal matrixのcondition numberが大きい
- raw donor driftのweighted MADが大きい
- smoothed donor driftのweighted MADが大きい
- raw/smoothed local-linear推定の差が大きい

validation wellと同じexp226 source foldはdonorから除外する。近傍wellのtrue error、符号付きbias、K12/K24 disagreement、GR likelihoodはriskに入れない。

## 検証方針

### Stage 0: donor-support risk readout

予測値は変更せず、K16 segmentごとに次を比較する。

```text
p_base  = 0.50 p_exp226 + 0.25 p_likPF + 0.25 p_exactHMM
p_other = 0.50 p_likPF + 0.50 p_exactHMM
benefit = RMSE(p_base) - RMSE(p_other)
```

riskと`benefit > 0`のAUCを読み、pooled `>=0.60`、4/5 foldsでAUC `>0.5`、within-well circular controlとの差`>=0.05`、top risk decileの平均benefit`>=0.10 ft`、top-bottom差`>=0.25 ft`をすべて要求する。1つでもFAILなら同じOOFで設定を救済せず閉じる。

### Stage 1: bounded weight shrink

Stage 0全PASSと別承認後だけ、次の1式を評価する。

```text
a_j    = 0.25 × clip((risk_j - 0.80) / 0.20, 0, 1)
delta  = clip(a_j × (p_other - p_base), -5 ft, +5 ft)
p_new  = p_base                              (md_since < 250 ft)
p_new  = p_base + delta                      (md_since >= 250 ft)
```

最大時でもexp226の重みは`0.50 -> 0.375`にしか下がらず、likPF/exact-HMMは各`0.25 -> 0.3125`となる。hard routingや補正方向推定は行わない。

## 判定

- exp263比RMSEを`0.02 ft`以上改善し、4/5 folds改善。
- activated subsetを`0.10 ft`以上改善。
- 0--250 ftはbitwise parity。
- 1000+、hidden-like 2面、by-well p95は非悪化、worst wellは`+0.25 ft`以内。
- 実gateのgainが同数circular controlを`0.02 ft`以上上回る。

## 実行量と境界

Stage 0は1 risk + 1 deterministic control、773 wellのsupport再構成、model/booster/HMM/PF/Beam 0。Kaggle CPU version 2で3,783,989 rows / 12,368 segmentsを209.829秒で完了した。Stage 1は未実装のまま閉じ、親再学習・予測再生成、inference、submissionは0。

## 所見

technical hard checksとcoverage checksは全PASSし、発火は762,529行（20.1515%）/ 433 wells / 5 foldsだった。一方、pooled real AUCは0.562091で0.60未満、controlとの差は0.005310で0.05未満、top-risk mean benefitは-0.674259 ft、1000+とhidden-like 2面の方向guardもFAILした。donor supportには弱いfold-stable signalがあるが、circular controlから分離できず、高risk側でもdestinationへの移動は平均悪化するため補正gateとして不採用とする。

## 既存枝との分離

- exp322はGRが弱い場所でexp263からexp226へ戻す設計だった。本実験はdonor supportが弱い場所でexp226から離すため、gateも方向も異なる。
- exp303で不支持だったK12/K16/K24安定性は使わない。
- exp279/281のようにHMMへhard routeまたは再decodeしない。

## 次

事前登録どおり救済gridを行わずbranchを閉じる。必須依存を満たさないexp330も未実装・未実行で閉じ、新しい同系救済backlogは追加しない。
