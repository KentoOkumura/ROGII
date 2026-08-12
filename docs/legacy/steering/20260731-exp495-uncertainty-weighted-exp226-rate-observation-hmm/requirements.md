# 要件

## 依頼

exp209 exact HMMで観測されたpersistent rate状態の追従遅れを、exp226の
fold-safe geometry rateを使って補う。ただしexp491のようにrate状態を除去して
遷移量をhard置換せず、既存rate transitionへ不確実性付きの弱い外部観測として融合する。
バックログ、実験ディレクトリ、steeringを作成して設計だけを確定し、実装は行わない。

## 制約

- Route: `pf_beam`。
- 親HMMは`exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`。
- 正の平均signalはexp355、主因診断はexp408、negative evidenceはexp411 / exp491を参照する。
- exp226 final`tvt_pred`の行差分は使わない。GR補正前のfold-safe`tvt_geop`由来U-rateだけを使う。
- exp209のposition/rate state、41 rate states、rate span、momentum、diffusion、TVT grid、
  position kernel、Gaussian GR emission、prior、forward-backward、posterior meanを固定する。
- 変更する科学的変数は、exp209 rate transitionへ掛けるexp226 rate Gaussian観測因子だけとする。
- 観測分散は既知prefixだけから作り、unknown suffix truth、error、fold outcome、persistent roleを
  schedule / confidence / candidate freeze前に読まない。
- rate観測のscale、temperature、clip、threshold、row/well gate、blend、selector、PFを追加しない。
- 親/control HMMは再実行せず、保存済みexp209 / exp355予測を比較に使う。
- Stage 0A、Stage 0B、Stage 1はそれぞれ別承認とし、自動昇格しない。
- inferenceとsubmissionは本設計の範囲外とする。
- 再現性は`docs/06_reproducibility.md`に従い、入力・schedule・confidence・predictionの
  logical / decompressed content SHAを記録する。

## 受け入れ基準

- exp495 scaffoldとsteeringが作成され、`config.yaml`にroute、lineage、式、段階、gate、
  実行量、禁止事項、承認lockが明記されている。
- Stage 0Aは0 HMM / 0 modelで、既知prefix rate残差scaleがsuffix rate誤差を
  target-freeに順位付けできるかだけを検証する。
- Stage 0Aが全gateを通過し、別承認された場合だけStage 0Bの1 variant × fixed32
  = 32 HMM well-runsを許可する。
- Stage 0Bがtechnical / mechanism全gateを通過し、別承認された場合だけStage 1の
  1 variant × 773 HMM well-runsを許可する。
- deterministic anchorとは扱わず、gzipはraw SHAではなくdecompressed content SHAを
  主証拠にする。
- notebook実装、Kaggle package、push、run、inference、submissionが未実施である。

## 2026-07-31 追加依頼

ユーザーの`exp495を実装してください`により、上記の設計のみ停止点からStage 0Aの
compact self-contained候補と契約テストの実装までを追加承認した。正規Notebook採用、
Kaggle package、push、run、Stage 0B / Stage 1、inference、submissionは追加承認に
含まれない。

## 2026-07-31 Stage 0B override

ユーザーの`Stage 0Bへ進んでください`を、Stage 0A mechanism gate FAILによる
fail-closed停止点の明示overrideとして記録する。Stage 0Bは事前登録済みのfixed32、
scientific variant 1、candidate HMM 32 well-runs、保存済みexp209 / exp355比較のまま
実装・Kaggle private CPU実行する。Stage 0Aの基準をPASSへ再分類せず、prefix window、
sigma式・floor、temperature、threshold、gate、emission、grid、blend、selector、PFを
変更しない。Stage 1への自動昇格は引き続き禁止する。
