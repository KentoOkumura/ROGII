# 設計

## アプローチ

exp417で凍結されたscale-5 likelihood-PFを単独出力するhidden-compatible
compact self-contained inference Notebookを、後続の実装承認後に1本だけ作る。
公開testではexp413 v4が保存した`likpf_scale_5`列と完全一致させ、hidden testでは
同じraw-test generatorをsample submissionのwell集合へ動的に適用する。

この実験はtrain-side不採用判断を覆すpromotion実験ではない。Public LB上で、
row-weighted OOF改善とby-well tail FAILのどちらが公開testに近かったかを記述する
late-stage censusである。

## 実験範囲

- 対象実験: `exp452_scale5_likpf_direct_public_lb_audit`
- Route: `pf_beam`
- 親実験: `exp417_scale5_seed_aggregation_promotion_audit`
- generator参照: `exp413_scale5_likpf_full_replacement_on_exp335` v4
- kernel参照: `exp072_exp063_full_replay_feature_cache`
- 変更する変数: なし。凍結済みscale-5候補を単体submissionへ露出するだけ。
- 固定する変数:
  - 500 particles / 128 seeds / seed index 0--127
  - `gs x1.0`、base sigma clip `[10,60]`、post-multiplier clipなし
  - temperature 5、full-suffix seed evidence
  - exp072互換dynamics、roughening、resampling、GR補完、typewell grid
  - float32 candidate surface
- 将来の実行Notebook:
  `exp452_scale5_likpf_direct_public_lb_audit_inference.ipynb`の1本
- template由来train Notebookは設計scaffoldであり、実行・実装対象にしない。
- train/model/booster/control rerun: `0 / 0 / 0 / 0`
- competition submission順: 3候補中1番目。LBを見て順序や式を変えない。

## 固定候補

1. 各wellについてexp072互換PFを128 seed生成する。
2. 各seedの全未知suffix GR log-likelihoodを固定する。
3. `w_s = exp((loglik_s - max_s loglik_s) / 5)`を計算し正規化する。
4. 行ごとに128 seedのTVTを`w_s`で平均し、float32の
   `likpf_scale_5_x1p0`とする。
5. この列をそのまま`submission.csv`の`tvt`へ書く。

公開parityの正はexp413 v4
`artifacts/exp263_replay_for_exp145.csv.gz`の`likpf_scale_5`列であり、
logical content SHAは
`b713ade7adb5b185dacc941edf19aec324bcd7e075a8e903d33a23f59eb809f3`とする。

## LB評価契約

- OOF evidence:
  `11.594897884 -> 10.914522073`、gain `0.680375810 ft`、5/5 folds。
- tail evidence:
  by-well p95 `+2.941688483 ft`、worst `+25.311274575 ft`。
- Public LB control:
  exp434のSHA256-seed `likpf_mean`再提出結果を使う。seedが異なるexp069
  Public LB `9.721`は直接controlに使わない。
- `candidate LB < control LB`なら公開splitではscale-5集約を支持、
  それ以外は不支持と記述する。
- Kaggle表示3桁の同値はtieとし、差を過剰解釈しない。
- 結果にかかわらずtrain-side tail FAILは履歴として維持し、自動採用しない。

## 再現性設計

- seed policy:
  `SHA256(well immutable id, feature family, seed index, base seed 42)`。
- stochastic処理の有無: PF particle generationにあり。
- PF/Beam / likelihood-PF / seed baggingの有無:
  likelihood-PFのみ。Beam、ML seed baggingはなし。
- 並列処理と乱数の関係:
  well/seedごとに独立local RNGを作り、global RNGを使わない。
- CPU/GPU runtime:
  CPU-only、internet off。GPUなし。Kaggle上限30,600秒。
- train cache / test regeneration SHA:
  公開参照surface、raw-test regenerated surface、schema、row/well manifest、
  decompressed content SHAを保存する。
- model manifest:
  modelなし。model count/booster countが0であることをmanifestへ記録する。
- prediction / submission SHA:
  candidate prediction logical SHA、CSV SHA、submission SHAを保存する。
- Kaggle bootstrap:
  prepare後にbootstrap内configと正のconfigの候補ID、seed、particle、temperature、
  runtime、submission禁止フラグが一致することを確認する。

## 技術gate

- source/config SHAが設計値と一致する。
- sample submissionのID集合、行数、列、well集合を動的に解決する。
- candidate rowsがsample submissionと1対1で一致する。
- finite coverage 100%、fallback 0、duplicate ID 0。
- 公開testではexp413 v4 scale-5 surfaceとfloat32最大差0。
- candidate surface freeze後にのみsubmissionを作る。
- submit-check PASS前および別承認前にcompetition submissionしない。

## リスク

- リークリスク:
  full-suffix GRは利用可能だがsuffix TVT、error、fold、hidden-like roleは生成時に読まない。
- CV/LB不一致リスク:
  OOFは773 wells、Public LBは小さい公開splitで、tail wellの有無に強く依存する。
- ランタイム/メモリリスク:
  hidden well数は公開3 wellsより多い。exp413 v4のhidden完走を根拠にするが、
  30,600秒を超える見積もりならpushしない。
- 再現性リスク:
  seed導出や並列順がexp413と違うと別Monte Carlo realizationになる。
- ガバナンスリスク:
  Public LBを見たtemperature/seed/particle/weight変更はblind LB tuningになるため禁止する。

## 実行後の解釈と次のアクション

private CPU inference version 1は技術gateをPASSし、ユーザー外部提出ref `55149125`は
Public LB `8.797`だった。同じSHA256 seed familyのarithmetic LikPF `9.807`より
`1.010`改善してOOFと方向一致したが、exp417のtail FAILは維持する。
追加run、rerun、再提出、候補順変更、LB後のパラメータ変更は行わない。
