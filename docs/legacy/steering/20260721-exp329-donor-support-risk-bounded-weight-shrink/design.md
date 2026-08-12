# 設計

## アプローチ

exp226が実際に使ったsource-fold-safe K16 donor集合を再構成する。距離、`n_eff`、local-linear condition、raw/smoothed donor MAD、raw-smoothed推定差の6項目をouter-train empirical percentileへ変換し、その平均を再rankしてsegment riskを作る。符号は持たせない。

Stage 0では、exp263固定式と非exp226成分の50/50式のsegment RMSE差をlate readoutし、riskのselectabilityだけを検証する。Stage 0を通過した場合だけ、risk上位20%から連続的に最大25%のexp226寄与を非exp226成分へ移す。row移動は5 ftでcapし、最初の250 ftは完全不変とする。

## 実験範囲

- 対象: `exp329_donor_support_risk_bounded_weight_shrink`
- Route: `pf_beam`
- 親: exp263の保存`0.50*exp226 + 0.25*likPF + 0.25*exact-HMM`
- 変更: target-free donor-support riskと、そのPASS後の1 fixed arithmetic shrink。
- 固定: 3 primitive prediction、fold identity、exp226 donor生成条件、HMM/PF/Beam/model。

## 原因分離

exp322はGR likelihoodの弱さを使ってexp226方向へ戻した。本設計はdonor supportの弱さを使ってexp226から離れる。exp303で不支持だったK12/K24 instabilityは入力に含めない。exp324のHMM`sig_r,t`も変更しない。

## 再現性設計

real pathはRNGなし。exp263 readout fold、exp226 source fold、well、segment、donorをcanonical sortし、各support primitive、CDF、composite、activation、predictionのSHAを保存する。controlはwell単位のnon-zero circular shiftをSHA256で一意に決める。

## リスク

- CV fold除外で人工的にdonorが疎になる: exp226 predictionを生成したsource foldとdonor除外を完全再現し、source-foldとreadout-foldを別列で監査する。
- riskが単なる空間clusterを拾う: signed errorを禁止し、within-well circular controlとの差を要求する。
- exp226を減らしすぎる:最大25%、5 ft cap、near vetoで制限する。
- coverage不足: 5--30% rows、100 wells、4 folds未満なら`INCONCLUSIVE`として救済せず閉じる。
