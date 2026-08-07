# exp500_exp490_mean_reversion_residual_likelihood_pf 結果

## 結論

ユーザーの明示オーバーライドに基づき、Stage 0 FAILを維持したまま、変更なしの単一variantを
773 wells・3,783,989 suffix rowsでfull OOF実行した。candidate RMSEは
`8.813504627 ft`で、保存exp404 `10.914522073 ft`から`2.101017446 ft`改善し、
5/5 foldsと全固定scopeを改善した。

一方、by-well delta RMSE p95は`+6.653601019 ft`、worst wellは`389ae58f`で
`+46.154671032 ft`悪化し、事前登録したtail guard 2件をFAILした。technicalは18/18 PASS、
scientificは12/14 PASSだが全AND条件を満たさない。最終状態は
`stage1_fail_closed_under_override`、次のactionは
`terminal_close_without_same_oof_rescue`である。inferenceとsubmissionは行わない。

## Full OOF RMSE

| 比較対象 | RMSE (ft) | candidateの改善 (ft) |
| --- | ---: | ---: |
| exp500 candidate | 8.813505 | - |
| exp226 final | 9.427110 | +0.613605 |
| exp404 likelihood-PF | 10.914522 | +2.101017 |
| exp486 residual-PF | 11.139812 | +2.326307 |
| candidate + exp209 HMM 50:50 | 8.661349 | - |
| exp404 + exp209 HMM 50:50 | 10.084910 | +1.423561 |

candidate単体はabsolute上限`9.407110 ft`もPASSした。50:50は固定readoutであり、
blend選択や昇格には使わない。

## Fold別RMSE

| fold | rows | candidate | exp404 | 改善 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 742,514 | 8.844861 | 9.360014 | +0.515153 |
| 1 | 770,907 | 9.059118 | 10.979419 | +1.920301 |
| 2 | 746,011 | 9.210184 | 10.694277 | +1.484093 |
| 3 | 746,131 | 8.129743 | 10.747502 | +2.617759 |
| 4 | 778,426 | 8.778095 | 12.482449 | +3.704355 |

## Scopeとepisode

raw GR observed / missing、high missing、MD 1000+、hidden-like spatial、
hidden-like typewell-purgedの全6 scopeでexp404を`1.933--3.036 ft`改善した。

- exp408 episode SSE: `48.9337%`削減
- exp408 episode count delta: `-142`
- exp408 recovery delta 256 / 512: `+0.029781 / +0.042320`
- exp410 episode SSE: `52.5254%`削減

これらの平均・scope・episode改善は、事前登録したwell-tail安全性FAILを相殺しない。

## 実行

- Kaggle shard: `kentookumura/exp500-mean-revert-resid-likpf-full-shard0`--`shard3`、各version 1
- Kaggle merge: `kentookumura/exp500-mean-revert-resid-likpf-full-merge` version 3、id_no `129465486`
- runtime: private CPU、internet off、GPU off
- scientific variant: 1
- candidate: 773 PF well-runs、98,944 seed-well、49,472,000 particle starts
- control PF / HMM / Beam / LightGBM / booster / GPU再実行: 全て0
- shard wall: `7,366.20 / 7,777.18 / 6,406.65 / 8,465.69 sec`
- merge/evaluation wall: `956.79 sec`、peak RSS `5.072 GiB`

merge version 1はCSV readback hashの技術不整合、version 2はexp226比較列の技術契約違反で停止した。
PF shardは再実行せず、version 3で保存artifact SHAと正しいpost-freeze `tvt_pred`参照を検証した。

## Technical / leakage / reproducibility

18 technical checksは全PASSした。4 shard artifact SHA再検証、773 wells / 3,783,989 rows、
5 folds、有限性、weight normalization、K16 half-life、saved参照RMSE parity、runtime / RSSを確認した。
union freeze前のtruth / control / role-fold-episode / forbidden geometry readsは全て0だった。

- scientific contract SHA: `dc5c1690312d76964bf4c0dbbb406402509049fa6828c44f6e42f13c0dea2c91`
- prediction logical SHA: `a4bfa0c48203566be31cfefa4c255182c0bec5949056d6ae688b5252b965210a`
- gate report SHA: `b9ff9832584d384498be191714241df58119db5462e9a04c284398d7a73b59d5`
- summary SHA: `019e730b6fc7d23017ac681f9a3c0ac4bb39b6673165433bd05808ba82e48680`

ログに結果と全artifact SHAがあるため、Kaggle output archive全体は取得していない。
最終確認に必要な小さいsummary、gate、primary metricsだけをversion 3から取得した。

## 解釈と次

K16 half-life平均回帰は、PFのpersistent basinだけでなくfull OOFの平均、全fold、全scopeでも
有効だった。しかし改善はwell間で不均一で、少数wellの大きな悪化を防げない。
平均RMSEだけなら強い候補だが、固定tail safety contractに従い本branchは終端閉鎖する。
同じOOFを使うhalf-life / noise / temperature探索、adaptive gate、blend / selector救済、
inference、submissionは行わない。保存full OOF artifactだけを使う原因readoutは、別の必要性と
承認がある場合に限り低優先度候補として扱う。
