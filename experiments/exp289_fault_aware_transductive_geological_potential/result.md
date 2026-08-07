# exp289_fault_aware_transductive_geological_potential 結果

## 状態

Stage 0 fault-topology association readoutをKaggle CPU version 3で完了した。技術guardは全項目を通過したが、事前登録したAUCとSpearmanの科学guardに届かなかったため、契約どおりbranchを閉じる。Stage 1/2、inference、submissionは未実装・未実施である。

## 仮説

outer-train ANCCとhidden-like known prefixから作るtarget-free fault riskが、exp226の損失を支配するpersistent whole-well biasを識別できるなら、fault cutを許した共通2D scalar surfaceをStage 1で検討できる。

## 設定

- 親: なし。新規standalone physics family
- Route: `pf_beam`
- 検証: 保存済みexp226の5-fold GroupKFold by wellを固定
- Stage 0: 1 audit variant、0 ML config、0 trained fold、0 booster
- primary risk: `suffix_fault_risk_p90`
- risk: cross-well k=12 donor spread、trajectory jump、known-prefix misfitの固定robust transform
- outer-valid: `MD/X/Y/Z/TVT_input`だけを読み、formation 6面とtrue suffixはrisk SHA freeze後にだけ接続
- source ANCC欠損: 全行非有限wellだけfold source donorから除外し、部分欠損はfail-closed
- control: 保存済みexp226 OOFだけをpost-freeze readoutに使用し、再生成なし

## Kaggle実行

- kernel: `kentookumura/exp289-fault-aware-geopotential-stage0-train`
- id_no: `127879234`
- canonical successful run: version 3、private CPU、GPU off、internet off
- runtime: `241.548128`秒
- peak RSS: `693.191406` MB
- version 1: 全行非有限source ANCCの入力契約で停止
- version 2: pandasが異なる長さのper-well attrsをconcat時に比較して停止
- version 3: 上記2件のfold-safeな技術修正後に完走
- solver / model / booster fit: 全versionで`0 / 0 / 0`

## Stage 0結果

| 指標 | 事前条件 | 結果 | 判定 |
| --- | ---: | ---: | --- |
| `abs(exp226 bias)>=10` AUC | 0.65以上 | 0.570652 | FAIL |
| risk対`abs(bias)` pooled Spearman | 0.25以上 | 0.127885 | FAIL |
| 正方向fold数 | 4/5以上 | 5/5 | PASS |
| technical guard | 全項目 | 全項目PASS | PASS |

fold別AUCは`0.515410 / 0.538095 / 0.626333 / 0.562908 / 0.586947`、Spearmanは`0.079190 / 0.136424 / 0.108868 / 0.175217 / 0.128489`だった。方向は安定して正だが、whole-well large biasを識別する強さは不足している。

CV、Public LB、Private LBはない。Stage 0は予測モデルではなく、Stage 1へ進む前の反証監査である。

## 生成物検証

- input manifest: 774件
- graph manifest: 5 folds
- target-free node risk: 320,991行
- target-free well risk: 773行、primary risk finite coverage 100%
- exp226 bias readout: 773行
- 全行非有限ANCC source除外数: fold別`6 / 4 / 6 / 6 / 6`、延べ28
- formation identity overall: RMSE最大`0.007182`、最大絶対誤差`0.030000`、相関最小`0.997634`
- manifest 7生成物のraw SHAとcanonical CSV content SHAを取得outputで照合済み
- `submission.csv`なし

## 再現性

- kernel version: 3
- pushed config SHA: `40a8cdcab3a9a29307c60ba5cdbe4079232061c5cbe84a16c1d8c03bcb0b9899`
- pushed train source SHA: `20cdaf42259a77ef40d615c4a3fa5f12d2bdf763d1851a64741410442342bfd4`
- graph manifest frozen SHA: `7040f60dc907bbb5b8c6bb86a05448a9d087445a598c1e24e8477da191d155e0`
- node risk frozen SHA: `2f2a6320f83237d5b55f5a11dedb9c4adbf4c5ef093b1829be58aafd66cd85af`
- well risk frozen SHA: `2f0cef466c6ba469573f969d190af5c8bda9b509be68084958ec1d9c67ff061e`
- summary SHA: `6dcaac7dc05bfb33d4f899720db7209ab19d883a2da413a4b917989efa7925f8`
- contract SHA: `33adb1f5ce430fe5ab71deaaea68d164c24c85d1da2f1d4ef83760c2e59d53d1`
- RNG: なし。fold / well / row / edge順序を固定
- rerun: 未実施

## 解釈

fault-riskとexp226 biasの関係は5/5 foldsで同方向だったため、局所的なgeological inconsistency signal自体はある。ただしAUCと順位相関は事前下限から大きく不足し、exp226のrare whole-well biasをfault topologyだけで十分に説明できない。well p90 aggregation、edge threshold、formation面を同じOOFで探索するとposthoc rescueになるため行わない。

## 次

`failure_policy=close_without_edge_threshold_formation_or_risk_aggregation_grid`を適用し、exp289のStage 1 MAP solver、Stage 2 GR factor、raw-test inference、submissionを閉じる。物理routeの次候補は、fault仮説の救済ではなくknown-prefix内で直接識別性を監査するexp290とする。全体優先順位では、既存LB anchorのtail regressionを0 boosterで再検証するexp276を上位に維持する。
