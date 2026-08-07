# 要件

## 依頼

exp226 の周辺井戸 local-linear K16 rate 推定を、TVT を一度予測した後の
target-free な後処理へ変換する。保存済み exp413 Stage D OOF を基準予測とし、
各 outer-valid fold を一つの擬似 test batch として扱う。同じ batch 内の他井戸の
`予測 TVT + Z` から作った K16 rate field だけで自井戸を弱く補正し、OOFを評価して
実験を完了する。

本依頼の範囲は backlog、steering、実験 scaffold、設計記録の作成までとする。
実装コード、Jupytext source、Notebook 変更、Kaggle package、実行、推論、提出は行わない。

## 制約

- Route: `ensemble`。exp413 ML予測と exp226 由来の空間 K16 rate 後処理が最終予測へ
  本質的に寄与するため。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 親 exp413、control、PF/Beam/HMM、selector、LightGBM を再学習・再実行しない。
- 保存済み exp413 Stage D OOF と outer fold manifest を固定入力とする。
- foldごとに outer-valid wells の exp413 OOF予測を全件先に揃え、他の
  outer-valid wells の予測だけを donor とする。自井戸は必ず donor から除外する。
- outer-train / outer-valid の真の suffix TVT、ANCC、その他 train-only formation面を
  prediction生成へ使わない。真の suffix TVT はprediction freeze後のscoreだけに使う。
- target well と donor well の raw `X/Y/Z`、row order、fold、exp413 OOF予測だけを
  prediction生成の allowlist とする。
- K16区間数、平滑化、local-linear KNN、support、補正率、clipを1契約へ固定し、
  同じOOF上でgrid、winner選択、gate救済を行わない。
- current public testは3 wellsしかなく、`min_unique_donor_wells=8`ではidentityになる。
  これはhidden約200 wellsを模したouter-valid fold評価を優先する意図的な契約であり、
  public 3-well出力を見てsupport条件を緩めない。

## 受け入れ基準

- steeringの仮説、入力allowlist、数式、固定parameter、validation、promotion gate、
  fail-close条件、実行inventoryが未確定項目なく記録されている。
- 実験ディレクトリはdesign-only scaffoldで、正規train/inference Notebookを
  template placeholderのまま保持している。
- 実装前の固定primaryは1候補だけで、raw exp413 control以外のreport-only candidate、
  parameter grid、target-derived selectorを含まない。
- 実装後に評価する場合は、3,783,989 rows / 773 wells / 5 folds、ID/order/fold/input SHA、
  prediction-before-truth freeze、self-donor exclusionを全てPASSする。
- primaryはexp413 CV 7.884802794404715に対してpooled RMSEを0.01 ft以上改善し、
  4/5 folds以上nonworse、固定MD/hidden-like scope、by-well tail、開始連続性の
  全AND gateを通った場合だけ採用候補とする。
- FAIL時はalpha、clip、K、bandwidth、rho、theta、support、fade、scope、gateを
  同じOOFで調整せず終了する。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。

## 次

設計確定後は停止する。ユーザーが実装を別途承認した場合だけ、固定済みの1 primaryを
compact self-contained train候補として実装する。
