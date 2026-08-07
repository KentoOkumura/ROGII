# 要件

## 依頼

`discussion711308_dz_dtvt_bpeak_cluster_baseline` backlog を実験化する。Kaggle discussion 711308 で報告された `dTVT ~= a*dZ + b` と `b` peak cluster の no-ML direct baseline を再現し、train pseudo-tail 診断と test submission 生成経路を同じ実験に入れる。v1 の rate-fit が Public LB 41.214 で要件未達だったため、v2 では row-step fit、X/Y/Z + last-300 TVT/Z clustering、prefix-holdout source selector まで含めて検証する。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- full train true TVT で fit した per-well `a,b` と full `b` peak label は診断と train source pool のみに使う。validation target well と test well の割当には tail true TVT を使わない。
- validation/test-like 割当は、known prefix、last-300 `TVT_input` / Z summary、X/Y/Z geometry、typewell exact hash、nearest train wells だけで行う。
- ML training、LightGBM config、GPU、control retraining は含めない。
- Public LB 約 12.8 の再現が目的であり、現行 final prediction への blend や direct adoption はこの実験では行わない。

## 受け入れ基準

- `.steering`、`config.yaml`、`SESSION_NOTES.md`、`result.md`、`metrics.json` が実験意図、コスト、再現性、次アクションを説明している。
- train notebook は full true `a,b` 診断、`b` peak distribution、target-free pseudo-tail CV、distance bucket、by-well、cluster purity、XY map artifact を生成できる。
- inference notebook は train full-fit source pool から同じ target-free assignment で test prediction と `submission.csv` を生成できる。
- 実装候補は at least `global_median`、`prefix_fit`、`peak_cluster_median`、`nearest_xy_k8`、`hybrid_peak_xy_k8`、`exact_typewell_peak_xy_k8`、step-fit feature-nearest、prefix-holdout selector を比較する。
- train-side CV と LB 約 12.8、exp113/117、exp148/198 ML anchor との比較観点が記録される。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。

## v4 追加要件

- selected inference は source / cluster `a,b` 選択ではなく、query/test well 自身の known `TVT_input` から `dTVT ~= a*dZ+b` を fit する。
- fit には unknown suffix の true TVT を使わず、全 known `TVT_input` rows のみを使う。
- unknown suffix は last known `TVT_input` を anchor とし、row step ごとに `a*dZ+b` を累積して予測する。
