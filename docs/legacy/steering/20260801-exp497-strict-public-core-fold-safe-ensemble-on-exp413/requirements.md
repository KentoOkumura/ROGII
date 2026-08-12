# 要件

## 依頼

公開上位 notebook の一般化可能な中核を、Public LB 固有の補正を除いた独立
public-core branch として再構成し、exp413 の保存済み OOF と予測レベルで
fold-safe に混ぜる実験を設計する。今回は backlog、steering、実験ディレクトリと
機械可読な契約だけを作り、実装、Kaggle package、学習、推論、提出は行わない。

## 仮説

exp413と同じfinal370面へmodel familyを追加するより、公開pipelineの一般化可能な
trajectory構造を独立させた方がresidual相関を下げ、constant blendで安全な改善を
得られる。Public LB固有処理を落とした状態で成立しなければ、公開scoreの改善は
coreの一般化性能ではなく公開surface適応に依存した可能性が高いと判断する。

## 制約

- Route は ensemble とする。独立 ML、PF/Beam、exp413 ML の全てが最終予測に本質的に寄与するためである。
- 親は exp413_scale5_likpf_full_replacement_on_exp335 とし、保存済み exp413 OOF、fold、scope、by-well を比較基準として再利用する。
- exp413 の selector 40本、signed selector 20本、TVT model 15本を再学習しない。
- public-core は exp413 final370、exp413 selector score、exp413 最終予測を特徴として使わない。両branchは最終 blend まで独立させる。
- Public LB、public test の well ID、row count、submission 値、公開提出 SHA を学習、選択、補正、fallbackに使わない。
- 同一well contact reconstruction、visible-prefix candidate calibration、公開well固定shift、Q0522、A27、model-package correction、precomputed submission fallback、public output copyを禁止する。
- 公開固定のwell-shape bin閾値とbin-to-variant mapは直接移植しない。outer-train wellsだけから学習し、outer-validへ適用する。
- 公開 pretrained boosterはfold provenanceが不足するためOOF生成に使わない。public-coreの学習器はouter/inner well分離で再学習する。
- final ensembleはOOF-level meta-fold cross-fitの単一constant convex weightだけとする。row/well gate、conditional router、worst-well ID ruleは使わない。
- 再現性は docs/06_reproducibility.md に従い、stochastic PF、GPU学習、Kaggle bootstrap、SHA記録を設計に含める。
- notebook実装は別承認とし、実装時はJupytext percent形式のcompact self-contained候補から始める。
- Kaggle train前にvariant、config、outer/inner fold、booster、PF/Beam run数を再計数し、ユーザーの明示承認を得る。

## 受け入れ基準

- strict public-core の採用部品、Public-LB特化として除外する部品、fold-safe境界が明示されている。
- exp413とpublic-coreのOOFが同じ3,783,989 score rows、773 wells、outer 5 foldsで一意joinできる。
- public-core単体、exp413単体、cross-fit blendを同じ行・fold・scope・by-wellで比較できる。
- primary blendはexp413比pooled RMSEを0.03 ft以上改善し、5/5 foldsでnonworse、全固定scopeでnonworse、by-well p95/worst deltaが各+0.25 ft以下である。
- meta-fold全てでpublic-core weightが0より大きく、public-core上限0.30を越えない。
- 上記AND gateを1つでも失敗した場合は、同じOOFでweight、bound、threshold、router、component、postprocessを救済調整せずbranchを閉じる。
- 実装前状態ではCV/LBを主張せず、notebookは実行不能なdesign placeholderのままである。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel versionを記録する。
- gzip生成物を比較する場合はdecompressed content SHAを主証拠として記録する。

## 2026-08-01 実装承認追補

ユーザーの「exp497を実装してください」により、Stage 0 compact self-contained preflightと
専用contract testの実装を承認済みとする。承認範囲はsource/parent SHA、outer/inner fold、
spatial pool除外、truth-late freeze、実行量/feature inventory、meta5 constant weight helper
までである。Stage P/M1/M2/E、正規Notebook採用、Kaggle package/run、inference、
submissionは別承認のままとする。

## 2026-08-01 実行承認追補

ユーザーの「実行してください」により、Stage P/M1/M2/E実装、正規train Notebook採用、
Kaggle package/runを承認済みとする。確定実行量は1 variant、2 branches、outer 5 / inner 4、
LGB 120 + CatBoost 80 = 200 boosters、Ridge 10、exp413再学習0である。ユーザーの明示指示に
よりColabは使わず、Stage MはKaggle GPUで実行する。inferenceとsubmissionは引き続き対象外。

## 2026-08-03 Stage I override承認追補

Stage E gate FAILとexp413選択をユーザーへ報告した後、ユーザーの「推論に進んでください」および
exp413再推論は不要との確認により、exp497不採用候補のprediction-only current-test inferenceを
検証目的overrideとして承認済みとする。train-side gate FAIL、selected anchor=exp413、Public-LB
昇格判断は変更しない。外部competition submitは対象外で、`submission.csv`も生成しない。

Stage Iは全train inner-4 OOFからfull-train deployment modelを1回作る。実行量は1 candidate、
2 branches、LightGBM 24 + CatBoost 16 = 40 boosters、Ridge 2、exp413再学習・再推論0。
既存exp413 current-test predictionをSHA固定入力として再利用し、public-coreとの最終blend weightは
Stage Eの5 meta-fold weight中央値`0.13716473330712417`に固定する。Colabは使わずKaggle GPUで
実行し、raw current testのrow/well数は動的に扱う。

## 2026-08-04 保存model推論実装承認追補

ユーザーの「推論作成に進んでください」により、Stage I version 4の保存済みLightGBM 24、
CatBoost 16、Ridge 2を読み込むhidden-safe推論専用候補の実装を承認済みとする。新規booster学習、
exp413再学習、weight再fitは0。public test固定のexp413 sidecarは禁止し、exp510 version 4で検証済みの
dynamic hidden-safe exp413 runtimeを同じ保存model/SHA契約で再利用する。dynamic sampleのID・well数を
固定せず、最終式は`0.8628352666928758 * exp413 + 0.13716473330712417 * strict_public_core`から
変更しない。推論NotebookはKaggle outputとして`submission.csv`を生成できるが、外部competition
submitは今回の承認に含めない。既存正規inference placeholderは上書きせず、まずJupytext起点の
`_compact_selfcontained_inference.py/.ipynb`候補を作る。
