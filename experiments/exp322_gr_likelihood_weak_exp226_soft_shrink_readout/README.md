# exp322_gr_likelihood_weak_exp226_soft_shrink_readout

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU train version 2完了、`INCONCLUSIVE_COVERAGE`、branch closed
- CV / Public LB / Private LB: `8.239202313 / なし / なし`
- Submit ID: なし
- 作成日: 2026-07-21
- 親実験: `exp263_last_anchor_better_candidate_confidence_pair_cache`
- shrink先: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`

## 仮説

GR shift likelihoodがflatでmodeを識別できない区間では、PF/HMM成分を含むexp263固定blendの根拠が弱い。ただしshift 0、すなわちexp226 K16が同じ尤度で棄却されていない場合だけなら、exp226へ小さく戻すことでtailを壊さず改善できる。

## 固定した予測式

```text
p_base = 0.50*p226 + 0.25*p_likpf + 0.25*p_exact_hmm

p_new = p_base
        + gate_H512
        * 1[md_since_last_known >= 250]
        * clip(0.25*(p226-p_base), -10, 10)
```

`gate_H512`は次の全条件で発火する。

1. exp280互換の13-shift GR emissionで、outer-train q20以下のtop1-top2 margin。
2. 同じscoreで、outer-train q80以上の正規化entropy。
3. shift 0 rankがtop3、またはbest-vs-zero gapがouter-train q20以下。
4. raw observed GR shareが`0.80`以上。

絶対GR尤度、final HMM posterior、exp133 ambiguous flagはgateに使わない。

## 検証方針

- Fold: exp263保存outer foldをreadout strataとして再利用し、exp226元OOF foldは別source identityとして監査。モデル学習や再分割なし。
- Block: 未知suffix先頭から非重複512行、short tail保持。
- Truth境界: score、gate、real/control予測をcontent SHAでfreezeした後だけtrue TVTを結合。
- 基準: exp263固定blend OOF RMSE `8.238331`。
- PASS: overall `>=0.02 ft`改善、4/5 folds、activated subset `>=0.10 ft`改善、near不変、1000+/hidden-like 2面/p95/worst非悪化、real gainがcircular controlより`>=0.02 ft`大きい。
- Coverage: changed row `1%--25%`、50 wells以上、4 folds以上。外れればPASSではなくinconclusive。

## 固定した禁止事項

- exp263/exp226/PF/HMM/Beam/K16の再生成・再学習。
- hard replacement、top1 shift correction、alpha/quantile/block/clip/emission grid。
- target/error/oracle gate、same-OOF rescue、ML selector。
- inference、submission。

## 実装

- `exp322_gr_likelihood_weak_exp226_soft_shrink_readout_compact_selfcontained_train.py`
- `exp322_gr_likelihood_weak_exp226_soft_shrink_readout_compact_selfcontained_train.ipynb`
- `exp322_gr_likelihood_weak_exp226_soft_shrink_readout_train.ipynb`（canonical採用済み）

compact候補をcanonical train Notebookへ採用した。train Notebookはexp263 cache、exp226 OOF、raw GR/typewell、hidden-like assignmentのhard guard、target-free freeze、late-truth metricsまでをself-containedで持つ。inference Notebookは変更していない。

## 実行状態

Kaggle CPU package/push/runは2026-07-21にユーザー承認済み。private CPU、GPU/TPU/internet offのversion 1（id_no `128089589`）はraw well scoring前に、exp226元OOF foldとexp263 readout foldの不一致を検出して停止した。両入力は別の保存済みgroup splitであり、親exp263との比較にはexp263 outer foldを使う。exp226元foldを各well一意のsource-fold identityとして別監査する最小修正後、version 2を195.332秒で完了した。

version 2の変更は4,870行、10 wells、5 foldsで、事前coverage下限の1%・50 wellsを満たさず`INCONCLUSIVE_COVERAGE`。RMSEはexp263 `8.238331715`から`8.239202313`へ`+0.000870598 ft`悪化し、activated subsetは`+0.688824530 ft`、改善foldは1/5、worst wellは`+0.261431339 ft`、real gateはcircular controlにも負けた。branchは救済なしで閉じ、inferenceとsubmissionは無効のままである。

## 所見

- exp280はGR shift signalの存在を支持するが、top1 `18.95%`のためdirect correctionは支持しない。
- exp281は常時offset decoderのtail riskを示した。本実験は常時利用せず、弱尤度かつexp226 admissibleなblockだけを対象にする。
- exp133はambiguity単独がbad regimeでないことを示したため、circular-shift matched controlを必須にする。
- 実測でもambiguity + shift 0 admissibilityは発火が狭すぎ、発火subsetを悪化させた。exp226へ戻す位置のtarget-free selectorとしては不採用。

## 次

exp322 branchは完了・不採用。alpha / quantile / block / clip / emission / ML selector救済、inference、submissionは行わず、新しい救済backlogも追加しない。
