# 設計

## CV 構造

outer well GroupKFold 5分割を評価単位とする。各 outer-train 内を inner well GroupKFold 4分割し、selector の outer-train OOF scoreを作る。同じ4本のinner modelで outer-valid を予測して平均する。これにより outer-valid の正解TVTは、selector feature生成にも最終モデル学習にも入らない。

## 特徴量

- selected/top1/top2 と last-known、exp218 prediction、主要candidateとの差
- predicted-error rank、top1/top2 margin、ratio、entropy
- source one-hot、candidate family
- segment length、switch、jump、boundary distance
- exact-HMM/self-GR-HMM/exp226 geometry confidence
- near-row risk、well-risk、candidate exclusion、fallback flag

## Safety gate

selector-only pathについて historical exp237 の guard を再現し、global/1000+/hidden-likeの改善だけで採用しない。near `000_050` のdeltaを0以下、worst-well最大回帰を0.25以下にすることを既定条件とする。不通過時は add-only最終学習を停止して診断生成物だけ保存する。

## 再現性

outer/inner splitはwellのsort順とGroupKFoldで固定する。candidate-long subsampleとLightGBM seedはouter/inner indexから決定する。upstream OOF、feature schema、fold manifest、model、predictionのSHAを保存する。PF/HMM/K16は固定生成物を読み、本実験で再生成しない。

## Notebook分割

selectorはCPU notebook、本学習はGPU notebookに分割する。selectorはfoldごとのtrain/valid predicted-error scoreをgzip CSVで保存し、decompressed SHAをmanifestへ記録する。本学習はguard pass、fold manifest、ID/well/row alignmentを検証してからのみ実行する。

## Raw-test推論分割

ユーザーがguard不通過とfold比較caveatを確認したうえで推論を明示承認したため、同じexp238内に二段推論を追加する。

- CPU selector trainはouter 5 × inner 4の20モデル本体、feature schema、outer/inner ID、best iteration、SHAを保存する。
- CPU `*_selector_inference.ipynb`: selectorを再学習しない。保存済み20本を読み、outer foldごとに対応する4本をraw testへ適用して、5個のfold-specific predicted-error score面を保存する。
- GPU `*_inference.ipynb`: exp218 raw-test 380特徴を再生成する。各保存済みLightGBMには同じouter foldのselector scoreから作った35 rank-slot特徴だけを渡し、15本を等重み平均する。
- exp109/114のOOF-only contextなどraw-test parityがない列は`NaN`のままとし、selector学習時と同じLightGBM native missing-value routingを使う。推論時の中央値補完や新規selector fitは行わない。
- PF/replayのraw-test生成物、selector score、prediction、submissionはdecompressed content SHAまたはfile SHAを保存する。推論notebookはcompetition submit APIを呼ばない。

このfold対応により、outer-validで使った「outer-train内の4 inner selector平均」とtest変換が一致する。全trainで新規4 selectorをfitして全15 LightGBMへ共通利用する経路は、nested学習時と特徴分布が一致しないため廃止する。

## Hidden-test code submission修正

提出ref `54647064` は、public test 14,151行向けのupstream inference生成物をhidden testへmergeしたため未処理例外になった。Kaggle code submissionではtest ID・well・行数が変わるため、別kernelで作ったraw-test row artifactは提出入力にできない。

- selector学習20本とfinal LightGBM学習15本は再実行しない。
- final submission notebookでcurrent testのpublic replay baseを一度だけ生成し、selectorとfinal modelの共通入力にする。
- exact/self-GR HMMは保存済みsource/configを使ってcurrent test上で決定的に再実行する。
- exp226 K16は保存済みpublic submissionを読まず、source-portをcurrent train/test上でfull-train inferenceする。
- exp145 learned-likelihoodは保存済みexp111 model/schemaをcurrent-test candidate/multiobs surfaceへ適用する。
- outerごとの4 saved selector平均はメモリ内で計算し、同じouterの3 final LightGBMへ直ちに渡す。public-test selector score CSVは監査用比較に限り、提出経路では使用しない。
- current sample submissionとのID one-to-one、全特徴有限、model schema、model SHAを提出生成前に検証する。
