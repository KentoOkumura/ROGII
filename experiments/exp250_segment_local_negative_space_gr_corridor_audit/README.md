# exp250_segment_local_negative_space_gr_corridor_audit

## 状態

- ルート: PF/Beam (`pf_beam`)
- 状態: Stage 0 manual parity PASS、Stage 1完了、guard 2/8 PASSで不採用
- decision: `fail_close_segment_local_hard_use_and_grid_search`
- CV / Public LB / Private LB: 対象外
- 推論 / 提出: 無効
- 親: `exp246_negative_space_gr_barrier_audit`
- fixed candidates: `exp072_exp063_full_replay_feature_cache`

## 仮説

well全tailへ履歴を累積したexp246とは異なり、GR mismatch topologyをMD 256 ft segmentごとにresetし、左から右へ連続するlow-mismatch corridorだけを見ると、良い候補を長期に巻き込まずにbad candidate eventを濃縮できる可能性がある。

## 実装

- MD 256 ft、stride 128 ft、horizontal 4 ft/bin。
- flat-Z prior中心のtypewell ±256 ft / 4 ft / 129 states。
- well-wide separate robust-z normalization。
- real GRとstable circular-shift shuffled typewell GRのpaired control。
- 右向きDAGのminimum-bottleneck pathと`tau_star + 0.25` corridor。
- first anchored / later spanning graph、segment間historyなし。
- fixed 5 candidatesのcorridor risk、truth coverage、overlap、hidden-like、by-well guard。

## 検証方針

Stage 0ではraw-onlyで選んだ12 wellsのfirst/middle/last plotについて、MD/TVT軸、4 ft grid、flat-Z center、real/shuffledのsupport/source/candidate parity、synthetic DAG/DP contract、plot/source SHAを確認した。manual parity通過後だけ同一configでStage 1の773 wellsを処理し、pooled/family AUC、q90 lift、good false-alert、overlap、hidden-like、1000+、by-well、truth coverageの事前固定8 guardを評価した。

## 実行

Stage 0 version 1でsynthetic/input contractと36 PNG / 72-row paired manifestを確認し、manual parityをPASSした。同じ科学設定のStage 1 version 2で773 wells / 3,783,989 candidate rowsを処理した。Stage 1 runtimeは7,633.823秒、model config / fold / boosterは0 / 0 / 0だった。

## 主結果

- pooled real AUC: 0.530134
- real - shuffled AUC: +0.035934（PASS）
- q90 bad-rate lift: 0.776971x
- q90 good false-alert: 0.232020
- overlap path差: median 57.61 ft / p90 258.684 ft
- overlap risk Spearman: 0.448723
- hidden-like AUC: 0.531044 / 0.532323
- by-well good false-alert: p95 0.757381 / max 0.984733
- truth coverage real - shuffled: overall +0.059580（PASS）

8 guardのうちreal-shuffled AUC差とtruth coverage差だけがPASSした。識別力、q90 lift、good false-alert、overlap、hidden-like、by-well安定性はFAILだった。

## 所見

0–100 ftではreal AUC約0.82が見えるがshuffled AUCも約0.77で、支配的な1000+ bucketのreal AUCは0.515575だった。nearの見かけのsignalはdistance / base-error構造の交絡を強く含み、globalまたはfamily横断のcandidate riskとして使えない。

exp246のglobal hard-history結果と既存exp249の結果・実装は、exp250のsegment-local minimum-bottleneck signalとして再利用していない。exp249は本実験から分離したまま変更していない。

## 実行入口

- train: `exp250_segment_local_negative_space_gr_corridor_audit_train.ipynb`
- inference: `exp250_segment_local_negative_space_gr_corridor_audit_inference.ipynb`（train-side-only guard）
- 実行環境: Kaggle CPU notebook、GPU/internet disabled

## 利用判断

不採用。candidate変更、hard prune、window統合、threshold/slack/segment grid、HMM/PF/Beam edge cut、`topk_path_confidence_features`への追加、raw-test inference、submissionへ進めない。再訪は保存済み成果物によるnear-distance交絡の低優先readoutに限定する。

## 次

本実験は完了。新規実行は行わない。将来再訪する場合も、保存済みcandidate-segment artifactを使う距離条件付きattribution readoutだけを低優先で行い、新しいcorridor計算やparameter探索へ広げない。
