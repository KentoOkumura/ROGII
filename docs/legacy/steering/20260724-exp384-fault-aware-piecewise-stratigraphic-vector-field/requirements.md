# 要件

## 依頼

`exp383_all_tvt_stratigraphic_vector_drift_field`が大幅改善した場合に、
単一の滑らかなvector fieldでは平均化される断層・地質domain境界を、
outer-trainの6地層面と正解TVTからpiecewise fieldとして表現する。

exp383の全TVT surface/catalog/field/path契約を固定し、smooth base fieldと
複数domain fieldを不確実性付きで周辺化する。物理モデル単独LB 6.5ロードマップのP1とする。

2026-07-24のユーザー指示でimplementation-onlyへ進める。compact self-contained
train候補、正規train Notebook、fail-closed inference候補、専用contract testを実装する。
exp383未実装・未PASSのため、Kaggle package/push/run、科学score、推論、提出は行わない。

## 制約

- Route: `pf_beam`
- 親: `exp383_all_tvt_stratigraphic_vector_drift_field`
- exp383 Stage 0/1 PASSはKaggle実行のhard prerequisiteとして維持する。今回の直接指示は
  コード実装と正規Notebook採用だけの承認で、run unlockとは扱わない。
- exp383のfold、全TVT windows、surface、signature、prefix校正、exp226 fallback、
  path solverを変更しない。
- fault/domainの教師にはouter-trainの正解TVTと6地層列を使える。
- target domain推論では`MD/X/Y/Z/TVT_input`とouter-train由来surface/domainだけを使う。
- outer-valid/testの生Formation、suffix TVT、error、oracle domainを使わない。
- hardな単一domain選択を行わず、smooth baseとpiecewise domainを固定posteriorで周辺化する。
- fault threshold、graph k、component size、posterior temperature、base floorを同一OOFでgridしない。
- ML、HMM、PF particle、Beam、GR/typewell likelihoodは含めない。
- 公開3 sample wellsでdomainやthresholdを決めない。

## 受け入れ基準

- steering、config、README、SESSION_NOTES、result、metricsに
  implementation-onlyかつexp383 artifact待ちの契約が一致している。
- compact self-contained train/inference sourceと正規Notebookがあり、trainは
  fault graph、component field、posterior、path、Stage 0/1をセル上で追える。
- exp383 Stage 0/1 PASS manifestとlogical SHAが未固定ならrun前にfail-closedになる。
- role guard、fixed AND cut、stable component、planar field、base floor、no-component
  exact fallback、prefix hard constraint、late truth join、disabled inferenceの専用testが通る。
- `KAGGLE_DIRECTION.md`でexp383 PASS条件付きP1として記録される。
- Stage 0でdomain graph、fault edge、component、target posteriorをtruth前にfreezeする。
- eligible query coverage`>=0.80`、domain posterior finite coverage`1.0`、
  primary componentのunique donor wells p05`>=12`を満たす。
- Stage 1でexp383比`0.50 ft`以上、4/5 folds、1000+とhidden-likeを改善する。
- no-fault/low-eligibility位置ではexp383へ数値的に戻る。
- exp383 controlを再実行せず、保存済みprediction/SHAを使う。
- deterministic anchorはrerun SHA一致まで主張しない。
