# 要件

## 依頼

地層面を個別に距離補間する方式から離れ、outer-train全坑井の正解TVTと6地層面を
共通の相対地質時間（RGT）へ変換する。全坑井の対応関係をcycle-consistent graphとして
同時に解き、target坑井について8〜32個の物理的に異なるTVT pathを生成する。
今回はバックログ、実験scaffold、steeringと設計だけを確定し、実装・実行は行わない。

2026-07-24 の後続ユーザー指示 `exp386を実装してください` により、別名compact
self-contained train候補、fail-closed inference候補、専用testの実装までを承認範囲へ
追加した。正規Notebook採用、Kaggle package/push/run、inference、submissionは引き続き
未承認とする。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md`に従い、graph/node/path順、SHA、Kaggle bootstrapを設計に明記する。
- exp383/384を親にせず、独立したtopology-first RGT familyとする。
- outer-valid/testは`MD/X/Y/Z/TVT_input`だけをscenario生成へ使う。
- targetの`GR`はexp386では読まず、exp387だけへ予約する。
- outer-valid/testの生Formation 6列とsuffix正解TVTはscenario-bank SHA freeze前に読まない。
- 6地層順、64 ft node、32 ft stride、outer-train q05--q95 stretch、
  近傍24 unique wells、scenario数8--32、0.5 ft diversityを固定する。
- target TVT予測をoracleで選ばず、exp386の出力は固定scenario bankとprior costだけにする。
- graph / edge / stretch / scenario count / diversity grid、ML、HMM、PF、Beam、
  親control再実行、current-test、inference、submissionは禁止する。

## 受け入れ基準

- Stage 0で3,783,989 rows / 773 wells / 5 folds、source-valid overlap 0、
  target GR/valid Formation/valid suffix truth read各0を満たす。
- RGT source coverage `>=0.95`、graph query coverage `>=0.95`、
  scenario-bank well coverage `>=0.98`、scenario count p05 `>=8`、
  finite path coverage `1.0`、cycle residual p95 `<=0.10 formation interval`を満たす。
- 16-well resource auditからfull runtime `<=30,600 sec`、peak RSS `<=25 GB`と見積もれる。
- 512-row prefix rolling-originでscenario oracleが保存済みexp226より`>=0.50 ft`改善し、
  4/5 folds以上で正である。
- truth-late scenario-bank oracle RMSEが`<=5.50 ft`、全5 fold正、
  bank coverage `>=0.98`、異なるscenarioを持つwell率`>=0.50`である。
- fold、RGT node/edge、cycle basis、scenario path、reference-GR template、
  role read ledger、logical/decompressed content SHAを保存する。
- deterministic anchorとして扱う場合は、graph/scenario/prediction content SHAと
  Kaggle kernel versionが成功rerunで一致している。
- gzip生成物を比較する場合は、raw `.csv.gz` SHAではなくdecompressed content SHAを主証拠とする。
