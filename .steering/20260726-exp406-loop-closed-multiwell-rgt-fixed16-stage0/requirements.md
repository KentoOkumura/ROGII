# 要件

## 依頼

exp405が技術的に有効なscientific FAILとなった場合だけ、
loop-closed multi-well RGTの実現可能性を固定16 wellsのStage 0で検証する。
exp386のFormation-derived RGT graphを救済するのではなく、current-testでも観測可能な
horizontal GRからpairwise局所対応を作り、その相対TVT offsetを閉路整合する
独立したGR-first topology familyとする。

exp405判定後のユーザー承認により、今回はfixed16 Stage 0のself-contained
Jupytext候補と専用testまでを実装する。Kaggle package/push/run、正規Notebook
採用、16 wellsを超えるfull run、suffix prediction、current-test、inference、
submissionは別承認なしに行わない。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 開始条件は`technically_valid_exp405_scientific_fail`だけ。exp405のtechnical ERROR、
  未実行、部分結果では開始しない。
- 固定16 wellsはexp386と同じexp226 outer-fold identityとround-robin
  `(fold, sorted well_id, offset)` selectorを再利用し、結果を見て選び直さない。
- outer-valid targetは`MD/X/Y/Z/GR/TVT_input`だけをStage 0 graphへ使う。
  suffix TVT、error、Formation 6列、oracle、hidden-like roleはfreeze前に読まない。
- outer-train donorは同foldのvalid wellを除外し、donor TVT/GR/geometryだけを使う。
- pairwise局所対応はH256 window、H128 stride、candidate-centered`±55 ft`、
  5 ft刻み、same-Type-Well優先の12 donor wellsへ固定する。
- graph solverはfundamental-cycle residualをHuber IRLSで同時に閉じる。
- Stage 0はgraph support、negative-control separation、prefix rolling-origin、
  resourceだけを評価し、unknown suffix predictionを保存しない。
- exp386のneighbor/stretch/Huber/scenario-count閾値を緩める救済、
  raw/solved residualの単位を混ぜること、ML/HMM/PF/Beamは禁止する。
- prefix512 controlはユーザー承認済みの推奨案として、target-free graph freeze後に
  exp226 original K16 donor field / adaptive Kappaをouter-trainからfold別に再構築し、
  fixed16 pseudo-cutの`tvt_geop`相当だけを生成する。このcontrolに限り
  outer-train ANCCをlate-readできるが、target ANCC、GR correction、U-projection、
  official OOF再生成は禁止する。

## 受け入れ基準

- exp405のscientific FAIL decision SHAを入力manifestで確認する。
- fixed16がexp386 selectorと一致し、5 reporting foldsすべてを含む。
- target/source overlap、suffix truth read、target Formation readはfreeze前すべて0。
- 16/16 wellsでgraph queryを作り、finite loop-closed RGT coverage `>=0.95`、
  graph query coverage `>=0.90`、connected target coverage 1.0を満たす。
- fundamental cycles `>=30`、solved cycle residual p95 `<=5.0 ft`、
  rawからのp95 reduction `>=50%`を満たす。
- real pairwise GR scoreがcircular controlよりpooledで`>=0.10`良く、
  4/5 folds以上で良い。
- known prefix末尾512 rowsのrolling-origin RMSEが保存済みexp226より
  `>=0.25 ft`改善し、4/5 folds以上で改善する。
- full 773-well Stage 1投影runtime `<=30,600 sec`、peak RSS `<=25 GB`。
- compact self-contained train / fail-closed inference候補がJupytext round-trip、
  `py_compile`、Ruff F821/F811、専用test、strict experiment validationを通る。
- すべてPASSした場合だけ、同じexp406内のfull-OOF Stage 1設計を
  別承認で追記できる。1条件でもFAILならexp406を救済せず閉じる。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
