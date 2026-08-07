# exp303_exp226_multiscale_k_stability_selectability_readout_on_exp264

## 状態

- ルート: `ml_model`
- 状態: Kaggle private CPU version 1完了・technical PASS / scientific FAIL・branch閉鎖
- CV: diagnostic pooled H512 AUC `0.488805`（AUC>0.5は1/5 folds）
- Public LB: なし
- Private LB: なし
- Submit ID: なし
- 作成日: 2026-07-20
- 親実験: `exp264_exp263_candidate_confidence_dual_selector`
- 候補source: `exp302_exp226_multiscale_k_segment_candidate_audit`

## 仮説

K12/K16/K24予測のlevel、H128 slope、segment boundary近傍jumpのtarget-free不安定性が、corrected
exp264 Stage C v6 hard selectorがK16を過小評価するH512 blockを識別できる。

## 開始条件

- exp302 technical PASS（充足済み）
- exp302 candidate novelty PASS（充足済み）
- corrected parentでのexp276再検証完了（充足済み）
- exp276 promotion guard FAIL（充足済み）

exp302 novelty FAILなら閉鎖する。exp276がPASSした場合は、ユーザー再承認なしでは実装しない。

## 変更点

- モデルを学習せず、1つの固定target-free scoreだけを診断する。
- scoreはlevel spread、H128 slope spread、boundary jump spreadのouter-train percentile平均。
- primary unitは非重複H512 block、scoreはblock内row p90。
- labelは`RMSE(K16)+0.25 <= RMSE(exp264 selected hard)`。
- truth/error/oracleはfeature freeze後のreadoutにだけ使う。

## 検証方針

- Fold: corrected exp264 Stage C v6のouter 5 folds
- Group: `well_id`
- Primary unit: 未知suffix先頭originの非重複H512 block
- Primary metric: fixed instability scoreによるpositive labelのROC AUC
- Leakage check: feature/schema/score/blockをtruth-freeでSHA freezeし、別loaderでtruthを後結合

## 固定PASS条件

- pooled H512 AUC `>=0.65`
- 4/5 foldsでAUC `>0.5`
- top/bottom quintile positive-rate lift `>=1.5x`
- top/bottom quintile mean K16 benefit差 `>=0.25 ft`
- 1000+とhidden-like 2面で方向非回帰

PASSしてもこのexpではselector、gate、predictionを作らない。別のstrict nested add-only selector-feature実験の
設計根拠に限定する。

## 実行契約

- 1 fixed readout × 5 evaluation folds
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- candidate regeneration / parent retraining: `0 / 0`
- CPU、推論なし、提出なし

## 実装

- `*_compact_selfcontained_train.py`をJupytext percent形式、18 cellsのself-contained readoutとして実装した。
- K12/K24はexp302 freeze manifestの固定SHAと宣言content SHA、保存gzipのdecompressed SHAを二重検証する。
- K16は保存済みexp226 OOF、selected hard/fold/MDはcorrected Stage C v6 candidate-longを必要列だけ読む。
- unique ID文字列を全件保持せず、well code/row indexへstream alignmentしてCPU memoryを抑える。
- row feature、outer-train empirical CDF、row score、H512 blockをparquet/JSONへ保存してSHA freezeする。
- freeze検証後の別loaderだけがraw TVTを読み、positive labelと固定readoutを作る。
- 1000+ scopeは`H512 block内の最小MD >= 1000 ft`、segment boundaryは各Kのcontinuous `linspace` edge±8行に固定した。
- inference notebookはfail-closedで、selector/gate/current-test prediction/submissionを生成しない。

ユーザーの実行承認後、別名compact版を正規train/inference Notebookへ採用した。trainはprivate CPU
`kentookumura/exp303-k-stability-readout-train` version 1で実行し、承認消費後はlocal packageの
`train_run_on_push=false`へ戻した。inferenceはfail-closedのまま実行していない。

## 結果

Kaggle version 1（id_no `128080983`）は約142.125秒で完了し、feature coverage、重複block 0、
truth-before-freeze 0、固定input SHA、score再計算を含むtechnical checksは全PASSした。

| 指標 | 結果 | 判定 |
| --- | ---: | --- |
| pooled H512 AUC | `0.488805` | FAIL（`>=0.65`未達） |
| AUC `>0.5` fold | `1/5` | FAIL（`>=4/5`未達） |
| top/bottom positive-rate lift | `0.916190x` | FAIL（`>=1.5x`未達） |
| top-bottom mean K16 benefit差 | `-1.205532 ft` | FAIL（`>=0.25 ft`未達） |
| 1000+ / hidden-like方向 | `0/3` | FAIL |

全metricsと生成物SHAはKaggle logへ表示されているため、大きなoutput archiveは取得していない。

## 所見

- exp300のoracle診断を直接featureにせず、raw-test生成可能なK-scale情報だけへ落とした。
- exp276 corrected-parent version 3は固定guard FAILとなり、既存risk familyでは解けないことを確認した。
- exp302 technical/noveltyとexp276 completion/FAILの4 dependencyはすべて成立した。
- 固定instability scoreはpooledでランダム近傍よりわずかに逆方向で、fold4以外の4 foldsと全stress scopeも逆方向だった。
- technical contractは成立しているため、入力不整合ではなくK-scale instability feature familyのscientific FAILと判断する。

## 次

同じOOFで方向反転、feature weight、horizon、boundary幅、thresholdを救済せずbranchを閉じる。
新規のK-stability救済expは追加せず、独立したexp305 exact-HMM emission auditを優先する。

## 表記

用語は`KAGGLE_DIRECTION.md`の表記方針と`docs/glossary.md`に合わせる。
