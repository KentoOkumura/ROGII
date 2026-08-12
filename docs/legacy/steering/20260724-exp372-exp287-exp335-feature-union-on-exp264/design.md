# 設計

## 仮説

exp287とexp335は同じcorrected exp264を親とし、pooled CVはほぼ同等だが、優位なfoldが異なる。
exp287 formation 74列はfold 1/4、exp335 signed residual 23列はfold 0/2/3で相対的に良い。
両familyを同じdownstream LightGBMへadd-onlyで渡せば、単独親では利用できなかった相補性を
学習し、best standalone CVを超えられる可能性がある。

ただし、両実験は同じworst well `fb03ae90`を悪化させた。平均改善だけで昇格させず、
exp264基準のtail guardを独立した必須条件として保持する。

## 実験範囲

- 対象実験: `exp372_exp287_exp335_feature_union_on_exp264`
- Route: `ml_model`
- 科学的親: `exp264_exp263_candidate_confidence_dual_selector`
- formation親: `exp287_fold_safe_formation_74_addonly_on_exp264`
- signed親: `exp335_signed_residual_meta_on_exp264`
- 変更する変数: downstream feature surfaceへformation74とsigned23を同時追加する。
- 固定する変数:
  - corrected exp264 outer 5 fold、well group、score rows
  - target residual from last known TVT
  - clean273とsaved compact74
  - exp287で生成済みのfold-safe formation74
  - exp335で生成済みのstrict-nested signed23
  - exp218/exp063 LightGBM config 0/1/2
  - seed 42、GPU mode、early stopping、thread数
- ユーザー判断により、別0-booster相補性診断は本実験の先行条件にしない。

## 根拠

| 指標 | exp287 | exp335 | 読み方 |
| --- | ---: | ---: | --- |
| pooled CV | 8.136708220 | 8.146107756 | exp287が0.009400 ft良い |
| Public LB | 7.530 | 7.517 | exp335が0.013良い |
| 優位fold | 1, 4 | 0, 2, 3 | fold相補性がある |
| worst well | fb03ae90 +8.228410 | fb03ae90 +10.238752 | tail失敗は共通 |

Public LBは仮説の背景としてのみ保持し、特徴・parameter・gate選択には使わない。

## 入力契約

### exp264 surface

- clean feature: 273列
- saved nested compact: 74列、25 partition
- compact manifest SHA:
  `f4855726de446b8308a8acf80d6ff6cd6a789f18ef90e165b98fa05d12aecf1c`
- compact schema logical SHA:
  `23614916c99edbbd513bcefee958d26cdfae5b83fb05c232c19736f2708dd725`
- saved OOF SHA:
  `b11c5005ca566f76588f4e1735386c15b8f016b874701a82e1c0741c8b839ae2`

### exp287 formation surface

- feature: 74列
- partition: outer 5 × train/valid = 10
- formation manifest SHA:
  `25611e281299991d626f1caca48673aee6225a890ad47ecdcd28a117ae827772`
- relationship audit SHA:
  `868cc2bc3d8ea57103c70a2c150f240a29cc4d0087595d9fc4d68e864f0c86a3`
- saved OOF SHA:
  `8f026c5c5f6508fb142981832994c6ba9cded4940168c648a9df9f3e698c3913`
- train roleはouter-train referenceから対象well自身を除外、valid roleはouter-train
  referenceだけで生成済み。exp372では再生成せず、manifest内のfile/logical SHAを検証する。

### exp335 signed surface

- feature: 23列
- partition: outer 5 × inner/valid role = 25
- signed compact manifest SHA:
  `237486930a0e6f7479d40d2b2d2ccb8e033e3787eb273c406d1eb5a3fc8a6b64`
- signed schema logical SHA:
  `74abf31f057dfe29177221895e3e26c5a261e5b51defc04f081d6b140f2be44c`
- Stage S reproducibility manifest SHA:
  `f4e37ce4bb8f38c2e9abc462b4625965612ac36815068cf9360766fff5d70ccb`
- saved OOF SHA:
  `8b28a3f29b981cbba118c9f98a5e7dd92e75613d87dddce39c2d162fb6a769b1`
- outer-trainはinner OOF、outer-validはouter-train内4 inner model ensembleの保存値を使う。
  exp372ではsigned selectorを再学習しない。

## 444特徴の組み立て

各outer foldで次を行う。

1. exp264 compactのtrain/valid roleを読み、`id, well`順をcanonical indexとする。
2. exp287 formationの同一outer/roleをmanifest SHA検証後にexact joinする。
3. exp335 signed compactをexp335 Stage Dと同じstrict-nested規則でtrain/validへ組み立て、
   exact joinする。
4. target/errorを開く前に、列名、列順、重複、row/well/fold/role、partition SHAをfreezeする。
5. 特徴順を
   `clean273 -> saved74 -> formation74 -> signed23`
   に固定し、444 unique columnsを要求する。
6. fit/early stopping用targetを読み、3 configを学習する。

saved exp287/exp335 OOF predictionは比較用であり、特徴へは入れない。exact duplicateや高相関は
reportだけ行い、pruneしない。

## 学習契約

- active variant: 1
- variant: `formation74_signed23_union_addonly`
- LightGBM configs: 3
- folds: 5
- 合計GPU boosters: `1 × 3 × 5 = 15`
- exp264 control再学習: 0
- exp287/exp335 standalone再学習: 0
- parent/signed selector再学習: 0
- formation/signed train feature再生成: 0
- runtime: Kaggle T4、internet off、`gpu_use_dp=true`、`deterministic=true`、
  `force_col_wise=true`、threads 8

実装、正規Notebook採用、package、runはそれぞれ別承認を必要とする。設計時点でKaggle GPU
quotaがないため、将来はquota回復かColab実行のどちらかを明示決定する。

## Gate

### Technical

- 3入力manifestと全partition file/logical SHA一致
- 3,783,989 rows / 773 wells
- id/well/fold/role完全一致、outer train-valid overlap 0
- schema freeze前truth/error読込0
- 444 unique features、model matrix finite
- 15 unique `(fold, config)` model slots

technical FAILはmodel fit前に停止する。同じ科学契約のtechnical retryにも別run承認を求める。

### Incremental utility

- pooled CV `<= 8.116708220359452`
  - best standalone exp287 `8.136708220359452`から`>=0.02 ft`改善
- foldごとの`min(exp287, exp335)`比delta `<=+0.02 ft`を4/5 folds
- near / mid / 1000+ / hidden-like spatial / hidden-like typewell-purgedの各scopeで、
  それぞれのbest standalone比delta `<=+0.02 ft`
- formation familyとsigned familyのtotal gainがともに正
- 各追加familyが4/5 folds以上でpositive gain

### Tail promotion

- exp264比by-well delta p95 `<=0.0 ft`
- exp264比worst-well delta `<=+0.25 ft`
- clean273比`+1/+3/+5 ft`悪化well数`<=135/39/14`

technical、incremental utility、tail promotionの全条件をANDで通した場合だけtrain-side promoteする。
平均改善だけを理由にtail FAILを上書きしない。

## Failure policy

- technical FAIL: fit前停止。仮説を変えないtechnical修正だけを候補にし、retryは別承認。
- incremental/tail FAIL: branch close。
- 同じOOFでfeature除外、family weight、config、threshold、gateを選ばない。
- exp287/exp335 prediction blend、hard selector、postprocessへ自動展開しない。
- inferenceとsubmissionは結果にかかわらず別明示承認が必要。

## 再現性設計

- seed policy: exp264 fold seedとLightGBM family seedを固定し、新規feature生成乱数を持たない。
- stochastic処理: 将来のGPU LightGBM学習だけ。
- PF/Beam / likelihood-PF / seed bagging: 新規実行なし。保存compactの由来としてのみ存在する。
- 並列処理: feature生成を行わず、global RNGやthread scheduling依存の乱数消費を追加しない。
- GPU: bitwise reproducibleとは扱わない。deterministic flagsと全model SHAを記録する。
- feature SHA: manifest、各Parquet file SHA、logical float32 content SHA、444列schema SHA。
- model/prediction: 15 model manifest、各model SHA、OOF SHA、metrics SHA。
- inference/submission SHA: 本設計の範囲外。将来別承認で実施した場合だけ記録する。
- bootstrap: 将来push直前にpackage notebook、embedded config/support、metadataのSHAと
  kernel sourcesをpull-backで照合する。

## リスク

- リークリスク: 3 surfaceのfold/roleを誤結合するとstrict nested契約が壊れる。
- 相関リスク: 共通親由来のため、97追加列が重複signalとなり増分が小さい可能性がある。
- tailリスク: 両単独親が同じworst wellを悪化させており、unionで悪化が増幅し得る。
- メモリリスク: 444列のfold matrixは421/370列親より大きい。float32、chunk copy、
  fold単位解放を固定し、全fold同時保持を禁止する。
- CV/LBリスク: exp287/exp335のCV/LB順位が逆転しており、0.01 ft級差は分布差やnoiseの
  可能性がある。Public LBでparameterを選ばない。
- 再現性リスク: GPU LightGBMはbitwise一致を保証しない。

## 実装対象外

- Jupytext source、helper、tests、正規Notebookの実装
- Kaggle/Colab package作成、push、run
- inference current-test feature生成、saved model inference、submission
- 0-booster OOF blend診断
- feature/config/gate rescue

## 2026-07-24 実装承認後の境界

上記のdesign-only境界のうち、Jupytext train候補、保存feature union pipeline、
専用contract test、静的validationだけを実装対象へ変更する。正規Notebook採用、
Kaggle/Colab package、push/run、inference、submission、rescueは対象外のまま維持する。

pipelineはtarget/errorを開く前にclean273 allowlist、exp264 saved74 schema、
exp287 formation74 schema、exp335 signed23 schemaを
`clean273 -> saved74 -> formation74 -> signed23`の順でfreezeする。その後だけclean baseと
保存OOFを開き、各foldで3 surfaceを`id/well/fold/role`完全一致で結合する。

## 2026-07-25 推論override設計

trainの科学FAILをPASSへ再分類せず、保存済みmodelのcurrent-test predictionだけを例外的に作る。
実績あるexp335 CPU inferenceを構成基準とし、exp287のall-train-reference formation生成を
追加してunion modelへ渡す。

### 推論処理

1. exp263固定契約からraw testの12 candidateと21 native-confidence列を再生成する。
2. 保存済みexp264 Stage C selector 40 modelで、outer foldごとのsaved compact74を作る。
3. 保存済みexp335 signed selector 20 modelで、outer foldごとのsigned compact23を作る。
4. exp218 current-test 380列をraw testから再生成し、clean273 allowlistへ固定する。
5. 773 train wellsをreferenceに、target test formation列を読まずformation74を1回生成する。
6. 各outer foldで
   `clean273 + saved74[outer] + formation74 + signed23[outer]`を444列順に組み立てる。
7. exp372 version 2の3 configs × 5 folds = 15 saved modelをCPU predictorで読み、
   residual predictionを等重み平均して`last_known_tvt`へ加える。
8. prediction、feature schema、formation content SHA、model audit、metrics、
   reproducibility manifest、`submission.csv`を保存する。

### 実行量

- raw-test candidate families: 12
- saved parent selector models: 40
- saved signed selector models: 20
- saved union TVT models: 15
- formation current-test generation: 1
- fitted model / trained booster / control retraining: 0 / 0 / 0
- runtime: Kaggle CPU、internet off

### Fail-closed条件

- 固定authorization tokenまたは`run_inference`がfalse。
- model bundle / manifest / schema / source catalogのSHA不一致。
- model count、outer/config coverage、feature order、444列countの不一致。
- raw-test candidate、compact、formation、signed、clean surfaceのID/order/finite不一致。
- sample submissionのrow/order/header/finite不一致。
- submit API authorizationがtrue。

### 再現性

- public-test保存row artifactは入力に使わず、raw testから同一runで再生成する。
- formation生成はRNGなし、sorted wellとquery workers 1を維持する。
- 保存modelは全SHAを検証し、prediction gzipはdecompressed content SHAも記録する。
- submission SHA、feature schema SHA、formation parquet/logical SHA、kernel versionを記録する。
- GPU学習済みmodelのbitwise再現性は主張せず、今回のCPU prediction artifactを固定する。
