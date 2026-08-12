# 要件

## 依頼

HMM / PF のGR観測尤度を単純な行別GR差や相関係数へ置き換えるのではなく、
known prefixから得るaffine不確実性とGR残差の系列相関を明示的に含む
block predictive likelihoodとして評価する。

`exp427_affine_ar1_whitened_gr_likelihood_readout`のbacklog、steering、
実験ディレクトリをdesign-onlyで作成し、仮説、固定式、入力境界、要因分解、
評価gate、再現性、禁止事項を確定する。実装、Notebook編集、Kaggle package /
push / run、HMM / PF decode、inference、submissionはまだ行わない。

## 2026-07-28 追加依頼

ユーザーからexp427の実装依頼を受けた。これによりStage 0 compact
self-contained train候補、専用contract test、fail-closed inference候補の実装だけを
追加承認とする。

既存正規Notebookの上書き・採用、Kaggle package / push / run、HMM / PF / Beam decode、
prediction、inference、submissionは追加依頼に含めず、引き続き未承認とする。

## 2026-07-28 実行依頼

ユーザーの「実行してください」により、compact self-contained train候補の正規
`*_train.ipynb`採用、Kaggle package、canonical train kernelへのpush、固定Stage 0
CPU runを追加承認する。

実行量はscientific primary 1、diagnostic ablation 2、matched control 1、
saved control 1、reporting fold 5、HMM / PF / Beam / model / LightGBM config /
trained fold / booster / GPU / parent control再生成をすべて0に固定する。
正規inference Notebook採用、prediction、inference、submissionは承認範囲外とする。

## 制約

- Routeは`pf_beam`。
- 科学的親と固定score surfaceは
  `exp280_exp226_shift_likelihood_separability_readout`とする。
- exp280のexp226 OOF path、13 shift、非重複512行block、tie order、
  truth-late境界、保存raw-Gaussian controlを変更しない。
- 主変更は、raw-finite GRに対するiid row-Gaussian aggregateを、
  prefix-posterior affine不確実性とouter-train AR(1)残差共分散を含む
  block Gaussian posterior-predictive log likelihoodへ置き換えること。
- 複合効果を原因分離するため、`identity/affine × iid/AR1`の固定2×2要因分解を
  明示的ablationとして使う。primaryは`affine_ar1`だけとする。
- Student-t / Huberのrow-emission直接置換、Pearson / ZNCC、
  SWT / DTW、Type Well群prior、GR imputation、transition / state / decoder変更を
  持ち込まない。
- candidate block scoreはraw finite GRだけで計算し、元のmissing位置をまたぐ
  AR innovationを作らない。
- affine posteriorはcurrent-well known prefixだけで更新し、affine事前分布は
  実行前の固定値、AR(1)係数はouter-train foldだけから固定する。
  outer-valid suffix TVT、error、formation、oracle shift、worst-well identityを
  score freeze前に読まない。
- Stage 0は0-HMM / 0-PF / 0-Beam / 0-model / 0-boosterの順位監査のみ。
- Stage 0 PASSでもdecoder実装やprediction変更を自動承認しない。別実験番号、
  新しいsteering、実行量提示、ユーザー承認を必要とする。
- 同一OOFでblock長、shift、prior、rho、clip、support、score weight、gateを
  grid探索しない。
- 再現性は`docs/06_reproducibility.md`に従い、input、score、eligibility、
  fold prior、manifest、metricsのcontent SHAを記録する。

## 受け入れ基準

- 2×2要因分解とprimary scoreの数式が一意に定義されている。
- affine prior、posterior update、sigma、AR(1)係数推定、finite run、
  eligibility、normalization、tie policyが固定されている。
- 保存exp280 controlとの比較と、同一raw-finite support上のmatched controlとの
  比較を分離している。
- technical / scientific AND gate、stress scope、negative control、
  FAIL時のterminal closeを事前固定している。
- Stage 0実行量がscientific score 3 + matched control 1 + saved control 1、
  reporting folds 5、HMM / PF / Beam / model / booster / GPU各0と記録されている。
- config、README、SESSION_NOTES、result、metricsをdesign-only状態で作る。
- `KAGGLE_DIRECTION.md`で既存P1/P2/P3と比較し、低-中P3へ配置する。
- deterministic submission anchorとは扱わない。
- gzip生成物を比較する場合はraw `.csv.gz` SHAではなくdecompressed content SHAを
  主証拠として記録する。
- compact self-contained実装と専用testが完了している。
- 正規train Notebook採用とKaggle Stage 0実行が明示承認されている。
- 正規inference Notebook採用・prediction・inference・submissionは承認範囲外である。
