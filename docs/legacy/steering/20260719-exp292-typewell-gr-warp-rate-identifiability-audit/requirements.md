# 要件

## 依頼

`KAGGLE_DIRECTION.md` の backlog `typewell_gr_warp_rate_identifiability_audit` を、
`exp292_typewell_gr_warp_rate_identifiability_audit` として設計する。

`exp268_multi_scale_initial_rate_candidates` が保存する `tail30 / w32 / w64 / w128 / w256`
の5本の exact-HMM rate candidateを固定入力とし、candidate pathでType Well GRをsampleした
forward GRとhorizontal GRの周波数・形状整合性が、true TVTを使わずに正しいrate candidateを
順位付けできるかをtrain-sideで監査する。

この段階ではsteeringとexperiment scaffoldだけを作り、notebook、実験ロジック、test、Kaggle
packageは実装しない。

## 制約

- Route: `pf_beam`。保存済み物理candidate bankのtarget-free識別性監査であり、ML学習やensembleは行わない。
- 科学的親は `exp268_multi_scale_initial_rate_candidates`。`exp209` exact-HMM GR emission、
  `exp288` Type Well / horizontal GR可視化、`exp170` / `exp211` affine negative evidence、
  `exp132` multi-scale GR scorer negative evidenceを固定参照する。
- `exp268` aggregateが、773 wells / 3,783,989 rows、5候補のid整合、candidate diversity、
  shard raw/decompressed SHA、aggregate content SHAをPASSするまでKaggle実行へ進まない。
- 入力candidateは `hmm_ir_tail30`, `hmm_ir_w32`, `hmm_ir_w64`, `hmm_ir_w128`,
  `hmm_ir_w256` の5本だけとする。候補追加、HMM/PF再生成、rate/window追加を行わない。
- known prefixの `TVT_input`, `MD`, horizontal `GR`だけでType Well GRのrobust affine calibrationと
  residual scaleをfitする。unknown suffixのtrue `TVT`、candidate error、oracle、fold metricを
  score、calibration、eligibility、shuffle、tie breakへ使わない。
- unknown suffixではhorizontal `GR`とcandidate pathから作るType Well forward GRだけを使う。
  score horizonは先頭から `H128 / H256 / H512`、primaryは `H256` に固定する。
- primary scoreはGaussian residual、NCC、chain-rule derivative residualの3成分をcandidate内で
  robust標準化し、等重みで平均する1式だけとする。score weight、horizon、calibration、thresholdのgridは行わない。
- negative controlはstable SHA256 local RNGによるwithin-well circular-shuffleだけとする。
  Python `hash()`、global RNG、処理順依存の乱数は使わない。
- no-GR geometry-only controlは、exp292のunknown-suffix GRでcandidateを選ばず、保存済み
  `hmm_ir_tail30`を常に使うsafe controlと定義する。upstream HMM自体のGR利用とは区別して記録する。
- Type Well範囲外を外挿しない。candidate共通paired coverage、Type Well local variance、
  forward derivative energyが不足するwell/horizonではGR selectionを無効にし、safe controlへfallbackする。
- target-free score / eligibility / top1 selection tableのcontent SHAを凍結した後にだけtrue TVTをjoinする。
- 1 audit variant、LightGBM config / trained fold / booster / HMM/PF well-runは `0 / 0 / 0 / 0`。
  評価用foldは5だが、model fitは0である。
- candidate平均、softmax TVT平均、top1 replacementの推論化、selector/weight学習、raw-test inference、
  submissionを行わない。PASSしてもsafe baseを必ず残すfollow-upを別途設計する。
- 再現性は `docs/06_reproducibility.md` に従い、Kaggle bootstrap、入力SHA、score/readoutの
  schema/content SHA、kernel versionを記録する。

## 受け入れ基準

- `config.yaml` に `experiment.route=pf_beam`、親/参照、5候補、3 horizon、primary H256、
  calibration、3成分score、shuffle、coverage、success guard、0-booster実行契約が明記される。
- exp268の2 shard、exp209 tail30、exp268 aggregate manifest/summaryをSHA付きでstrict alignし、
  id重複・欠損・候補欠損・well重複をfail-closedにできる設計である。
- target-free APIが `TVT`, `target`, `true_tvt`, `error`, `abs_error`, `oracle` をrejectし、
  score freeze前のtruth access 0をcontract testで確認できる設計である。
- H128/H256/H512ごとにGaussian residual、NCC、derivative residual、composite score、
  common coverage、calibration/variance/gradient eligibilityを候補別に保存する。
- primary H256のcandidate-best AUCをreal / circular-shuffleで同じ実装から計算し、
  pooled AUC lift `>= 0.02`かつreal-minus-shuffledが4/5 foldsで正なら識別性guardをPASSとする。
- fixed real-score top1がsafe tail30よりfull unknown suffix pooled RMSEを `>= 0.10 ft`改善し、
  4/5 foldsで改善することをselection guardとする。
- `1000+`、hidden-like spatial、hidden-like typewell-purgedでsafe tail30から非悪化することを
  subgroup guardとする。いずれかFAILならbranchをcloseし、同一truth上の救済gridは行わない。
- overall / fold / horizon / distance / prefix length / tail length / hidden-like / by-well / worst-well、
  candidate-best AUC、top1 RMSE、real-vs-shuffle、fallback reasonを出力する設計である。
- inference notebookはfail-closedで `submission.csv` を生成しない設計である。
- 実装時はJupytext percent source、正規ipynb、py_compile、Ruff F821、固有contract test、
  strict experiment validationを通す。
- 本実験はprediction/submission anchorではなく、固定入力へのdeterministic diagnosticとして扱う。
- gzip生成物はraw `.csv.gz` SHAではなくdecompressed content SHAを主証拠とする。
