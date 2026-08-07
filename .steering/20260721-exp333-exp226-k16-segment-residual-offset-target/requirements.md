# 要件

## 依頼

exp226とtruthの残差を1行targetではなくK16区間targetとして学習する案について、backlog、実験ディレクトリ、steeringを作成して設計を確定する。今回は実装、Notebook実行、Kaggle push、推論、提出を行わない。

## 仮説

exp226のrow residualは高分散だが、同じK16 segment内では主に低周波のdatum offsetとして共有される。exp226と同じK16境界で`mean(TVT - exp226_pred)`を1つのsegment targetへ集約し、target-freeなrow featureの区間平均から予測すれば、row-wise residual modelであるexp228より信号対雑音比を高められる。

## 制約

- Routeは`ensemble`とする。exp226のPF/Beam系base predictionとLightGBM residual offsetの両方が最終予測へ本質的に寄与するためである。
- 親は`exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`、row-wise比較は`exp228_direct_residual_correction_on_exp226`とする。
- 区間はexp226と数値互換のK16だけに固定する。H128/H256/H512、change point、可変長、window gridは同時評価しない。
- primary targetは各segmentのfloat64算術平均`mean(TVT - exp226_pred)`とする。median、Huber center、slope、curvature、row targetを同時に学習しない。
- segment sample weightはsegment row数とし、最終判定はsegment metricではなくrowへbroadcastしたOOF TVTのRMSEで行う。
- 補正は予測offsetをsegment全行へそのまま加える。clip、shrink、taper、interpolation、境界平滑化を行わない。
- featureはexp228/exp218系のうちraw-testで再生成可能な`projection_correction`、`u_disagreement`、`gr_wavelet_rotation_confidence`だけをrowからsegment finite meanへ集約する。supervisedな`learned_likelihood_confidence`、selector score、truth/error/oracle featureは使わない。
- exp226 baseはstrict nestedに作る。outer-train segment targetはouter-train内4-foldのinner OOF exp226 prediction、outer-validはfull outer-trainから生成したexp226 predictionを使う。
- outer foldは保存済みexp226 5-fold identityに一致させる。outer-valid exp226 predictionは保存済みexp226 OOFと最大絶対差`1e-8 ft`以下でなければ停止する。
- Stage 0は0 model/0 boosterの保存OOF headroom監査、Stage 1は1 variant × 1 LightGBM config × 5 outer folds = 5 CPU boostersに固定する。
- 保存済みexp226 OOFとexp228 OOF/metricsをcontrolとし、controlや親モデルを再学習しない。
- Stage 0実装、Stage 0 Kaggle実行、Stage 1実装、Stage 1 Kaggle実行、inference/submitはそれぞれ別承認とする。
- `docs/06_reproducibility.md`に従い、input、fold、segment assignment、feature schema/content、nested exp226 prediction、model、OOF predictionのSHAを記録する。

## 受け入れ基準

- backlog、steering 3文書、実験scaffold、config、README、SESSION_NOTES、result、metricsがdesign-only状態で整合する。
- K16境界、target、feature allowlist、nested fold、LightGBM config、実行量、promotion/停止条件が一意に定義される。
- Stage 0はK16 oracle offset-onlyが保存済みexp226を`1.00 ft`以上改善し、5/5 foldsで`0.50 ft`以上改善した場合だけPASSとする。
- Stage 1はexp228 `8.944085501`を`0.05 ft`以上改善し、exp226比4/5 folds改善、near/1000+/hidden-like/segment境界/by-well p95非悪化、worst-well`<=+0.25 ft`をすべて満たす場合だけ科学的PASSとする。
- Stage 1がPASSしても自動的に推論へ進めない。推論候補化には保存済みexp263 `8.238331715`以下と別承認を追加で要求する。
- configでimplementation、Kaggle run、inference、submissionがすべて無効である。

## 次

Stage 0は固定headroom gateをPASSし、32-well Kaggle CPU parity/runtime preflight version 1も親OOF parity最大差`1.819e-12 ft`、full runtime外挿`1.787 h`でPASSした。full Stage 1 Kaggle CPU version 1はCV`9.076676661`でexp226を改善したが、固定pooled上限未達、near 0--250悪化、worst-well`+8.099023 ft`により`FAIL_CLOSE_BRANCH`となった。実行承認は消費済みで、追加config、same-OOF救済、inference、submissionへは進まない。

## Stage 1実装承認追記

- `implementation.stage_1_enabled=true`はtrainコードの存在だけを示す。実行承認は`stage_1_preflight_approved` / `stage_1_run_approved`で別管理する。
- Stage 1 source / Notebookは別名候補として追加し、Stage 0正規Notebookを上書きしない。
- 重いexp226 predictorとGRWR generatorは固定bootstrap source dependencyとし、Notebook上ではsplit、feature allowlist、K16集約、学習、全gate、artifact/SHA orchestrationを追跡可能にする。
