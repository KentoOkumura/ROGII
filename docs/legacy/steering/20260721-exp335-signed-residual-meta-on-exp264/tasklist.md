# タスクリスト

## 比較元と変更点

- 比較元: corrected exp264 Stage C v6 / Stage D v3（clean273 + saved compact74 = 347特徴、保存済みcontrol）。
- 変更点: 既存74特徴を置き換えず、strict nestedのsigned-residual compact 23特徴だけをadd-onlyし、370特徴で後段TVTを比較する。
- 不変条件: control、fold、seed、候補順、selector入力88特徴、downstream 3 configを再学習・変更しない。

## 未着手（別途承認が必要）

- なし。

## 進行中

- なし。

## ブロック中

- なし。

## 完了

- [x] 仮説を「既存74列を維持し、候補誤差の方向情報だけをadd-onlyする」に固定した。
- [x] 親/controlをcorrected exp264 Stage C v6 / Stage D v3に固定した。
- [x] selector教師を`true_tvt - candidate_tvt`、objectiveを`regression_l2` 1 configに固定した。
- [x] selector入力88特徴、12候補、2 legal domain、outer5-inner4、seed 42を固定した。
- [x] 新規compactを候補別12 + 既存top-1注釈8 + 分布3 = 23列に固定した。
- [x] 既存74列の置換・削除・再学習を禁止し、saved control再学習0に固定した。
- [x] Stage Sを20 CPU selector boosters、Stage Dを15 GPU downstream boostersと記録した。
- [x] Stage S、downstream scientific support、train-side promotionのgateを分離した。
- [x] exp287/exp334、objective grid、hard selection、softmax/Viterbi、rolling/well集約を本実験から除外した。
- [x] `docs/06_reproducibility.md`に沿うSHA・GPU・Kaggle bootstrap契約を記録した。
- [x] 設計確定時点ではdesign-onlyで、実装・学習・推論・提出を行わない承認境界を記録した。
- [x] 追加の実装依頼をStage S実装だけの承認として記録し、実行権限とは分離した。
- [x] exp263 cache、corrected Stage A 88特徴、corrected Stage C v6 manifest/schema/25 partitionのSHA preflightを実装した。
- [x] `true_tvt - candidate_tvt`のformula parityとunavailable candidate fail-closedを実装した。
- [x] outer-train inner OOF / outer-valid 4-model ensembleのstrict nested L2 headを20-model契約で実装した。
- [x] 保存済み74列のscoreから既存top-1 identityを復元し、candidate別12 + top-1注釈8 + 分布3の23列を生成する処理を実装した。
- [x] Jupytext percent形式のcompact self-contained train候補と通常`.ipynb`を別名で作成し、正規scaffoldを維持した。
- [x] label、23列schema、parent key/top-1 parity、gate、Parquet chunk alignment、承認flagの専用testを実装した。
- [x] 検証済みself-contained候補を正規train notebookへ採用した。
- [x] Kaggle CPU version 2で0-booster preflightを実行し、25/25 parent partition SHAと1,024行top-1 parityをPASSした。
- [x] Kaggle CPU version 3で1 objective × outer5 × inner4 = 20 modelsを学習した。既存selector/control再学習は0。
- [x] Stage S technical gateをPASSした。20 models、25 partitions、18,919,945 compact rows、45,407,868 score rows、formula/top-1 parity `0.0`。
- [x] Stage S score gateをPASSした。pooled RMSE `8.430777`対prior `10.974123`、改善`2.543345`、5/5 folds改善。
- [x] small metrics/manifests/logをSHA付きで保存し、model/compact/score/reproducibility manifestを照合した。
- [x] Stage D実装と1 variant × 3 configs × 5 folds = 15 GPU boostersの明示承認を2026-07-22に得た。
- [x] saved exp264 control再学習0、clean273 + saved74 + signed23 = 370特徴のStage Dを別Jupytext notebookへ実装した。
- [x] pooled/fold/scope/by-well p95/worstとclean273比+1/+3/+5 ft件数の固定AND gateを実装した。
- [x] Stage D version 1の0-booster失敗を、clean273にも含まれる`last_known_tvt`の重複列選択まで特定した。
- [x] 順序保持de-duplicationと回帰testを追加し、370特徴・15 booster契約を変えずversion 2をcanonical T4 kernelへpushした。
- [x] Stage D version 2で15/15 GPU boostersを完了し、3,783,989 rows / 773 wellsのOOF、15-model manifest、fold/scope/by-well/importance readoutを生成した。
- [x] pooled RMSE `8.146108`、saved exp264比`-0.314703 ft`、4/5 folds、全5 scope改善を確認した。
- [x] by-well p95 `+1.728657 ft`、worst `+10.238752 ft`とclean273 promotion tailの悪化により固定gateをFAILと判定した。
- [x] small artifact SHAとpooled/fold/scope/by-well/promotion/signed-gain再計算を照合した。
- [x] gateを緩和せず、同一実験の救済、inference、submissionなしでbranchをクローズした。
- [x] ユーザー明示overrideに基づき、exp264 hidden-safe current-test regenerationを基にCPU self-contained inferenceを実装した。
- [x] Stage C 40 / Stage S 20 / Stage D 15 saved modelsと370-feature outer対応のfail-closed guardを実装した。
- [x] Jupytext / compile / Ruff / tests / strict validation / CPU metadataをPASSした。
- [x] canonical Kaggle CPU inference version 3をCPU / internet offで完了し、14,151 rows / 3 wellsを推論した。
- [x] sampleとのheader・行数・ID順、重複、NaN/Infを検証し、submit-checkをWARN/FAILなしでPASSした。
- [x] agentによる外部competition submissionを行わず、Stage D scientific-support / promotion FAILを保持した。
- [x] ユーザー実施のcode submission ref `54928806`がPublic LB `7.517`でCOMPLETEになったことを記録した。
- [x] Public LB anchor更新とtrain-side非promote判断を分離した。

## 現時点の結果

- Stage S完了・PASS。Stage D version 2は15 modelsを完了し、downstream CV `8.146108`を得た。平均は改善したがtail guard FAILのため非promote。別overrideのCPU inference version 3とsubmit-checkは完了し、user-submitted Public LB `7.517`を記録した。

## 次のアクション

- 非promote判断を維持し、Public LB `7.517`をsubmitted reference anchorとして扱う。
