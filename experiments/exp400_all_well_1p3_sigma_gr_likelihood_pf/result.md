# exp400 結果

## 状態

Kaggle private CPU version 1完了。technical gateはPASSしたが、
scientific promotion gateはFAILしたため、全well一律`gs × 1.3`
likelihood-PF branchを救済なしで閉じる。

## 仮説

exp072 likelihood-PFのalready-clipped GR観測scaleを全wellで1.3倍すると、
GR evidenceの過信とparticle mode collapseが弱まり、`likpf_mean`の
unknown-suffix RMSEが改善する。

## 固定設定

- parent: `exp072_exp063_full_replay_feature_cache`
- candidate: `gs_candidate = 1.3 * clip(gs_raw, 10, 60)`
- scope: 全773 wells、post clipなし
- PF: 500 particles、128 stable seeds、scale 3/5/8/12
- primary: arithmetic `likpf_mean`
- control: saved exp072、再実行なし
- reporting: 5 folds、fixed exp209-HMM 50:50 safety guard
- model / booster / HMM / Beam / control rerun: 0

## 実行結果

| 項目 | 値 |
| --- | ---: |
| Kaggle kernel | version 1 / id_no `128585102` |
| Runtime | 10,496.300秒（約2.916時間） |
| Rows / wells | 3,783,989 / 773 |
| Candidate RMSE | 12.221811 ft |
| Saved exp072 control RMSE | 11.594894 ft |
| Improvement | -0.626917 ft |
| Candidate MAE / bias | 7.442313 / -1.210092 ft |
| Within 10 ft | 0.758374 |
| Non-regressed folds | 1 / 5 |
| Improved / regressed wells | 305 / 468 |
| Fixed exp209-HMM 50:50 candidate / control | 10.659968 / 10.269693 ft |
| Public / Private LB | 未提出 / 未提出 |

fold別ではfold 3だけが`+0.099666 ft`改善し、fold 0/1/2/4は
`-0.511090 / -0.096551 / -0.423900 / -1.883659 ft`悪化した。

required scopeもすべてcontrolより悪化した。

| Scope | Improvement |
| --- | ---: |
| raw GR observed | -0.453077 ft |
| raw GR missing | -0.998656 ft |
| high missing-fraction wells | -0.884439 ft |
| suffix 1000 ft以上 | -0.708353 ft |
| hidden-like spatial | -0.706604 ft |
| hidden-like typewell-purged | -0.738688 ft |

by-well delta p95は`+5.059698 ft`、worst regressionは
well `708caea9`の`+32.160524 ft`で、固定tail gateもFAILした。

## Technical gateと再現性

technical gateはPASSした。

- 773/773 wellsをfallbackなしで完走し、finite coverageは1.0。
- 実行量は773 PF well-runs / 98,944 seed-well trajectories /
  49,472,000 particle startsで事前計画と一致。
- `gs_candidate / gs_base`誤差0、post-multiplier clip 0、
  candidate scale範囲`14.551610--78.0`。
- 保存exp072 controlとexp209 fixed blend controlのmetric parityはPASS。
- predictionはtruth join前にfreezeし、freeze前のtruth / fold / role readは0。
- prediction logical content SHA:
  `009a1d73e187c4126a70231214f14fbe1ae44edee47d9a166818ab1bd928a3bf`
- artifact manifest SHA:
  `59c877025e81713639c97822e14b6da1f77bee1d99274dc1f8e933d329ce8dfa`
- 小型8生成物はmanifest記載SHAと実ファイルがすべて一致した。
  86.8 MBのprediction本体は後続利用しないFAIL branchのため取得していない。

## Secondary scale診断

同一x1.3 PF runのscale 3/5/8/12 RMSEは順に
`11.271336 / 11.174615 / 11.243685 / 11.342899 ft`だった。
ただし保存exp072には対応するx1.0 scale列がなく、これらは事前固定どおり
candidate-only nonselective diagnosticである。実行後に最良だったscale 5を
primaryへ差し替えたり、x1.3の改善証拠として扱ったりしない。

## 解釈

全well一律のlikelihood scale拡大は、exp072 arithmetic seed meanを改善しなかった。
特にGR missing、高missing、fold 4、well tailで悪化が大きく、GR evidenceを一律に
弱めると一部wellの誤対応は救える一方、広い範囲で有効なGR拘束まで失うと考える。

探索的なwell-level readoutでは、改善率はmissing fraction下位四分位で51.0%、
上位四分位で24.4%、base `gs`下位四分位で56.2%、上位四分位で28.5%だった。
これは原因候補であり、同じOOFを見たadaptive multiplierやwell gateの根拠には使わない。

## 結論と次

decision:
`all_well_likelihood_pf_gs_x1p3_failed_close_without_rescue`

multiplier / clip / particles / seeds / scale / initial spread / resampling /
blend / selectorのsame-OOF救済、inference、submission、version 2は行わない。

原因確認が必要なら、保存済みwell auditとby-well metricsだけを使う
0-PF・0-modelのmissingness / base-scale / ESS・resampling
failure-attribution readoutを別設計にする。既存候補の優先順位は変更しない。
