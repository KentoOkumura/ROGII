# exp267 well-segment candidate divergence signature cluster

> exp263候補だけの18署名・structure negativeは保持。exp264 Stage B scoreを使う評価は無効。

## 状態

- ルート: `ensemble`
- 状態: Stage A Kaggle version 2完了・guard FAIL・branch closed
- CV / Public LB / Private LB: 対象外
- 作成日: 2026-07-17
- 親実験: `exp265_target_free_pairwise_candidate_divergence_soft_experts_on_exp264`
- candidate source: `exp263_last_anchor_better_candidate_confidence_pair_cache`
- post-assignment score: `exp264_exp263_candidate_confidence_dual_selector` Stage B version 2固定dataset

## 仮説

6 primitive candidateのwell内序盤・中盤・終盤における広がりを18次元target-free署名へ固定すれば、
exp265で失敗したblock-length clusterを使わずに、安定したlow/middle/high divergence well clusterを
作れる。そのclusterでcandidate winnerまたはexp264 expected-error calibrationが再現的に異なるなら、
exp264 dual selectorへwell署名をadd-onlyする価値がある。

## 変更点

- evaluation progressをfixed thirdsへ分け、6指標×3区間=18特徴をraw row-weighted集約する。
- raw MD span、block rows、absolute candidate TVTを使わない。
- outer-train median、RobustScaler、clip `[-10,10]`、KMeans K=3をfitする。
- low/middle/high soft membershipと別seed stabilityをouter-validへOOF付与する。
- assignment確定後だけ保存済みexp264 Stage B scoreをstreaming監査する。
- Stage Aは0 booster。conditional Stage Bの10 CPU boosters、inference、submissionはdisabled。

## 検証方針

- Fold: outer well 5-fold
- Group: `well`
- Structure: cluster occupancy、別seed一致率、valid側divergence profile、fold semantic整合
- Score: fold別candidate winner pattern、high-low calibration差、worst cluster leave-one-well-out
- Leakage: exp264 score、true TVT、error、oracleをcluster fitへ入れない

## 実行入口

- 学習/監査 notebook: `exp267_well_segment_candidate_divergence_signature_cluster_on_exp265_train.ipynb`
- inference notebook: disabled guardのみ
- 初回実行先: Kaggle CPU、GPU/TPU/internet off
- Kaggle kernel: `kentookumura/exp267-well-segment-divergence-cluster-train` version 2 / id_no `127573486`

## 現時点の判断

773 wells / 5 folds / 18特徴の技術契約、全wellの3区間coverage、別seed stabilityは通過した。
一方、middle clusterはpooled 41 wellsで基準75未満、3 foldsで10 wells未満だった。
low<middle<highのbank-range profileは1/5 foldsしか通らず、candidate winner patternもfold間で
一致しなかった。Stage A総合guardはFAILのためconditional Stage Bは学習せずbranchを閉じる。

## 所見

- 良かった点: exp265で問題になったraw MD / block-length proxyをfeature schemaから除外し、
  0 fallback、seed一致率min 0.954839を確認できた。
- 失敗原因: K=3 semantic clusterが不均衡で、18特徴全体のcentroid順序が区間別bank rangeの
  low/middle/high順序を安定して表さず、candidate winnerも再現しなかった。
- 注意: current-testは3 wellsなので、train-side clusterが安定しても提出一般化の根拠にはならない。
