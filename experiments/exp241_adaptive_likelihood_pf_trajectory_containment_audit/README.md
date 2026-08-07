# exp241_adaptive_likelihood_pf_trajectory_containment_audit

## 状態

完了・train-side不採用。Kaggle CPU shard 0/2/3の574/773 wells（74.3%）を監査し、
ユーザー判断によりshard 1は実行せず終了した。推論・提出は行わない。

## 目的

exp232/233 で sparse な target-free gate にもかかわらず約 +1.93 RMSE の悪化が出た原因を、
paired T=1/T=2 replay で診断する。well×seed の最初の gate 後に path divergence、ESS、
resampling、seed disagreement が有限 horizon に containment されるかを確認する。

## 仮説

exp232 の sparse gate 後の大幅回帰は、gate行単体のerrorではなく、conditional resamplingの
分岐とseed間のpath disagreementが後続行まで持続したために生じた。paired replayでevent後の
divergenceがlate horizonまで増えるなら、この仮説を支持する。

## 検証方針

well×seedの最初のtarget-free gateをeventとして固定し、paired T=1/T=2について
`8/32/64/128/256/512/1024/end` rowsのcumulative RMSE delta、path divergence、ESS、
resampling countを比較する。overall、1000_plus、hidden-like、worst-wellもguardとする。

## 固定条件

- route: `pf_beam`
- control: paired T=1 Gaussian likelihood-PF
- treatment: gate-on row だけ T=2
- 500 particles、128 seeds、seed mean
- horizon: `8/32/64/128/256/512/1024/end`
- stable SHA256 well moduloによる4 CPU shard
- LightGBM 0 config、fold 0、booster 0
- Kaggle CPU、internet disabled、inference/submission disabled

## 主要生成物

- event manifest
- event×horizon metrics と集約表
- row-level seed disagreement / ESS / resampling diagnostics
- candidate、distance bucket、hidden-like、by-well metrics

## 所見

T=2はpaired T=1よりoverallで`-0.011971`と微改善したが、hidden-likeとworst-wellの
guardを通らなかった。mean absolute path divergenceは8 rows後`0.127775 ft`からend
`3.123176 ft`へ増加し、trajectory containmentは支持されない。保存済みexp072と
再生成T=1の約+1.91 RMSE差も確認され、PF replay parityを分離せずにdirect likelihood
変更の効果を解釈できない。追加temperature/mixture grid、raw-test inference、submitへ進まない。

実行の正は `exp241_adaptive_likelihood_pf_trajectory_containment_audit_train.ipynb` とする。

## 次

なし。shard 1、追加grid、raw-test inference、submitは実行しない。
