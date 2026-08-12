# 要件

## 仮説

exp311で平均改善を示した同群priorに、target-freeなpeer/support availabilityと
fixed fallbackだけを加えれば、同群gainを保ちながら未知群・purged well・worst wellへの
negative transferを分離できる。

## 依頼

閉鎖済み `exp313_typewell_group_unseen_transfer_guard` の着眼点を、`exp311` / `exp312`
のpromotion PASSを要求しない独立した安全性readoutとして新番号へ切り出す。
保存済み `exp311` のType Well群統計に、事前固定したavailability / support / fallbackだけを
適用し、平均改善とworst-well悪化をtarget-freeな規則で分離できるかを診断する。

初回はbacklog、steering、design-only experiment scaffoldまでを作成した。2026-07-23の
追加依頼でStage 0 compact self-contained候補とcontract testsまでを実装する。
既存の正規Notebook採用、Kaggle package/push/run、後続ML/PF/Beam、inference、
submissionは今回も行わない。

## 制約

- Route: `pf_beam`。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- `exp313`をreopen / reparentせず、`exp311` Kaggle CPU v1の保存済み群統計とfoldを固定入力にする。
- `exp311` / `exp312` のpromotion FAILは上書きせず、本readoutがPASSしても旧exp314--320を自動解禁しない。
- availability、peer/support、fallback reasonはouter-valid truthやOOF errorを開く前に凍結する。
- exact groupはpeer wells `>=2`かつeffective rows `>=64`だけ利用する。
- fallbackは`exact group -> global prior -> identity/no-correction`に固定し、soft similarityは使わない。
- test相当で利用可能なのはType Well content、known-prefix GR、raw geometryだけとする。
- suffix TVT/error、formation train-only列、well ID rule、同一readoutでのthreshold/fallback救済は禁止する。
- Stage 0は3 audit surfaces / 5 folds / model・booster・decoder・HMM各0とする。

## 受け入れ基準

- fold-safe availability coverage `>=0.90`。
- identity fallbackのrow単位parity最大絶対誤差 `<=1e-10`。
- same-group held-out gain `>=0.05 horizontal GR API`、改善fold `>=4/5`。
- leave-one-group-outとspatial+typewell-purgedでnegative transfer
  `<=0.00 horizontal GR API`。
- worst-well regression `<=+0.25 horizontal GR API`。
- 全gate PASS時も得られるのは「固定guardに診断価値がある」という判定だけで、補正、selector、後続expの実装承認ではない。
- deterministic readoutとして扱う場合は、input/group/fold、availability/fallback table、score table、kernel versionのcontent SHAを記録する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。

## 実装時の単位訂正

exp311はTVT候補、予測器、decoderを生成せず、保存済みscoreはhorizontal GR API単位である。
補正・予測生成を禁止したまま保存値を監査するため、事前登録した閾値の数値は変えず、
誤っていた`ft`表記だけを`horizontal GR API`へ訂正する。

## 次のアクション

compact候補の正規Notebook採用とKaggle CPU Stage 0実行は、それぞれ別承認後に行う。
