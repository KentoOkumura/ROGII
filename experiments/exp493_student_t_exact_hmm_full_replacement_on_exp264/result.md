# exp493 結果

## 結論

Kaggle private CPU version 3で固定40 boosterのStage A/C、特徴量重要度、
科学readout、再現性summaryまで完了した。

hard primary RMSEは保存exp264の`8.652531956`から`8.616237400`へ
`0.036294555 ft`改善した。一方、改善foldは`3/5`に留まり、
by-well p95は`+0.540095855 ft`、worst well `f6d009f4`は
`+10.472288433 ft`悪化した。事前固定scientific gateはFAILで、
decisionは`FAIL_CLOSE_FIXED12_STUDENT_T_REPLACEMENT_SELECTOR`。

候補数を12に戻してもStudent-t置換の平均改善とwell-tail safetyは両立しなかった。
weight、threshold、domain、gate救済、downstream TVT、推論、提出へ進めず閉じる。

## 仮説

Student-t exact HMMを13本目として追加せず、Gaussian `exact_hmm` semantic slotと
全面置換して12候補を維持すれば、exp388で混入したcandidate count増加由来の
rerankingを除いてselector適合性を評価できる。

## 凍結した設定

- 12 candidate ID / order / domainを維持
- changed 4 / unchanged 8
- exp264 corrected Stage A 88列 / compact 74列を維持
- 1 variant / 2 objectives / outer 5 × inner 4 / 40 CPU booster
- 保存control再学習0 / GPU 0 / downstream 0 / inference 0 / submission 0

## Kaggle実行

- kernel:
  `kentookumura/exp493-studentt-fixed12-replacement-selector-train`
- `id_no`: `129218034`
- 成功version: `3`
- private CPU / internet disabled
- version 3 notebook elapsed: `5896.184330 sec`
- version 3 trained selector booster: `40/40`
- version 2とversion 3の累計CPU booster: `80`
- parent/control再学習: `0`
- full output archive: 未取得
- selected output:
  metrics、gate、fold、manifest、feature importanceなど小型成果物だけ取得

version 1は親config path未解決で`18.985 sec`、booster 0のまま停止した。
version 2は全40 boosterと科学readout後、feature-importance long-form CSVを
wide-formと誤認して`KeyError: Column not found: gain`で停止した。
ERROR runの成果物を回収できなかったため、ユーザー承認を受けてschema修正版を
version 3として追加40 boosterで再実行した。

## CV結果

| 指標 | exp493 | 保存exp264 | delta |
| --- | ---: | ---: | ---: |
| pooled hard primary RMSE | 8.616237400 | 8.652531956 | -0.036294555 |
| near 0--250 RMSE | 1.659804036 | 1.663644827 | -0.003840791 |
| distance 1000+ RMSE | 9.464135571 | 9.503798844 | -0.039663273 |
| hidden-like spatial RMSE | 9.431567982 | 9.536496454 | -0.104928471 |
| hidden-like typewell-purged RMSE | 9.344436064 | 9.412065207 | -0.067629142 |
| fixed fallback RMSE（report-only） | 8.160447731 | 8.238331546 | -0.077883816 |

fold RMSE / parent差:

- fold 0: `8.827610034` / `-0.164109991`
- fold 1: `8.547425036` / `+0.120790111`
- fold 2: `8.698265159` / `-0.202238182`
- fold 3: `8.566946237` / `+0.140473234`
- fold 4: `8.435310875` / `-0.064247797`

Student-t依存4候補のtop1は
`1,372,891 / 3,783,989 = 36.281580%`。平均改善は明確だが、fold 1/3と
一部wellの大きな回帰を相殺できなかった。

## Gate

PASS:

- 12候補順、4 changed / 8 unchanged、formula、availability/value parity
- global key join、missing key 0、source-fold feature利用0、truth-late
- Stage A 88列 / compact 74列schema parity
- 40 models、25 compact partitions、18,919,945 compact rows
- 45,407,868 outer-valid score rows
- nested leakage audit
- expected-error MAE、within10 logloss / Brierはprior比5/5 folds改善
- pooled、near、1000+、hidden-like 2面

FAIL:

- 改善fold: `3/5 < 4/5`
- by-well p95 delta: `+0.540095855 > +0.25 ft`
- worst-well delta: `f6d009f4 +10.472288433 > +0.25 ft`

## 特徴量重要度

両objectiveで`bank__candidate_mean_abs_disagreement`が最大だった。
`pred_abs_error`では`bank__candidate_abs_minus_median`、
`formula__component_std`、`conf__native__sigma_tvt`も上位に入った。
Student-t uncertaintyは利用されているが、現行hard selectorではwell-tailを
安全に抑える情報へ変換できていない。

## 再現性

- exp374 decompressed prediction:
  `668fe87da902955acee742c72d30724abb53f32050bb5d0a5c1b3dee0cbd626e`
- feature schema:
  `b91ec1517a82641fe4d96f41c97872151f273a8bbfcb537284f91d47aacf1035`
- compact schema:
  `e3a677610899cb33bf58262f4cf02f650300c8c2207c46b53588d3418162ea74`
- replacement semantic manifest:
  `a93d20c9cdc129c4ef8ec792d9beb80a55a54109cbd94858e878e799a43a7213`
- model manifest:
  `2560711a4a2f333a31cc2b3b423ace9d7b3d31062c0513e0505376e4ff80d0ca`
- outer-valid candidate score:
  `da4f6af616c3a87495d643cca3ef6e3e531cb2828dca769ef32fe7f4c6f9db89`
- compact manifest:
  `548514a2d775f42dbfc87a43f4e4d378c305b6f34d85722f10827ccc3e6fd5d9`
- scientific gate:
  `1c024bbe1893edb4afc96c1141c284e7f0ea74e1156b13908e947d4e82066bf5`
- summary:
  `d3811e4fe49f272d7b992fabee681b9e42ea8cd44efeba9295736ae033620b74`

version 2の成功成果物を回収できず、独立した完全rerun一致は確認していないため、
deterministic anchorには昇格しない。

## 次

hard candidate replacement/addition branchは閉じる。既存backlogの
`student_t_gaussian_disagreement_continuous_risk_feature_on_exp264`は、
Gaussian--Student-t disagreement / std / log-likelihoodの0-booster
target-free transfer readoutに限定し、別の必要性と承認がある場合だけP4で検討する。
exp493のsame-OOF救済や自動的なdownstream/inference移行は行わない。
