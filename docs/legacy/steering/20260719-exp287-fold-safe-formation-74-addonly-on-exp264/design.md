# 設計

## 仮説

exp218のformation系74列はfeature family自体が無効なのではなく、full-train referenceでOOFを作った
fit境界が無効だった。referenceをouter fold内へ移せば、修正版exp264へhidden-safeに再接続できる。

## アプローチ

修正版exp264のclean 273とnested compact 74は変更せず、availability auditで除外されたformation系74列を
fold-local referenceで作り直す。raw trainは一度だけreference catalogへ要約し、各outer foldでは
outer-train well集合を固定してplane / dense imputerを構築する。train roleの各target wellは自身の
reference rowsをquery時に除外し、valid roleにはouter-valid wellがreference catalogへ入らない。

5 fold × train/validの10 feature partitionをすべて生成し、欠損・nonfinite、schema、既存347列との
duplicate/correlation監査、feature content SHAを保存してからモデルfitへ進む。モデルはexp218/exp264と
同じ3 configとGPU reproducibility modeを使い、421列variantだけを各foldで学習する。controlは
corrected exp264 Stage D v3 OOF SHAを固定して読み、boosterを再学習しない。

## 実験範囲

- 対象実験: `exp287_fold_safe_formation_74_addonly_on_exp264`
- Route: `ml_model`
- 親実験: `exp264_exp263_candidate_confidence_dual_selector`
- 変更する変数: fold-localに再生成したformation系74列のadd-only。
- 固定する変数: Stage C v6 compact 74、clean 273、outer 5 folds、LightGBM 3 config、GPU mode、
  saved parent OOF、feature/grid、promotion guard。
- 学習量: 1 variant × 3 configs × 5 folds = 15 GPU boosters。
- 親/control再学習: 0 booster。
- inference / submission: fixed guardはFAILのまま。2026-07-20の明示overrideにより保存済み15 modelの
  inferenceだけを実行する。submission fileは形式検証用に生成するがcompetition submitは無効。

## データフロー

1. Stage C v6の25 partition、clean-273 allowlist、formation audit、corrected Stage D v3 OOFをSHA確認する。
2. exp218 surfaceからclean 273とmetadataだけをcopyし、旧formation列を破棄する。
3. raw train/current-test headerを監査する。current-test targetにformation列を要求しない。
4. raw train 773 wellsからplane medianとdense ANCC sampleのreference catalogを作る。
5. outer foldごとにouter-train referenceをfitし、train self-exclude / valid outer-train-onlyで74列を作る。
6. 既存347列とのexact duplicate、固定50,000行sampleのPearson/Spearmanをreport-onlyで保存する。
7. 全10 partitionのSHAと監査完了後にだけ、421列の15 boostersを学習する。
8. 保存済みexp264 347列OOFとoverall / fold / scope / hidden-like / by-wellを比較する。

## 再現性設計

- seed policy: formation生成はRNGなし。LightGBM family固有seedを維持する。
- stochastic 処理の有無: 新規formation生成はdeterministic。GPU LightGBMと保存済み上流compactは
  stochastic componentとして記録する。
- PF/Beam / likelihood-PF / seed bagging: 再生成しない。保存済みnested compactだけを入力する。
- 並列処理と乱数: target wellsをsort後にjoblib threadsへ渡し、cKDTree query workersは1。
  immutable idでcompact順へ再整列する。global RNGは使わない。
- CPU/GPU runtime: feature生成はCPU、学習はT4。`gpu_use_dp=true`、`deterministic=true`、
  `force_col_wise=true`、`n_jobs=num_threads=8`。
- train cache / test regeneration SHA: 各fold-roleでParquet file SHAとid+float32 logical content SHAを保存。
  current-testはguard PASS後にall-train referenceで別途生成し、同じschema SHAを要求する。
- model / prediction / submission SHA: trainでは15 model manifestとOOF SHAを保存。override inferenceでは
  prediction / feature schema / current-test formation / submission file SHAを保存する。
- Kaggle bootstrap: prepare後にmetadata、bootstrap内config、audit/allowlist/source filesを照合する。

## リスク

- リークリスク: self-exclusion漏れ、outer-valid reference混入、旧formation列の残存をfail-closedにする。
- CV/LB不一致: exp264はPublic LB anchorだがworst-well guard FAIL。exp287もtrain guardとLB判断を分離する。
- ランタイム/メモリ: 5 fold × 2 role × 74列cacheと421列matrixが大きい。base列をchunk copyし、
  foldごとにmatrix/modelを解放する。Kaggle初回実行で実測を記録する。
- 再現性: GPU bitwise同一は仮定せず、feature SHAとmodel/OOF SHAをversionごとに残す。
- 事後選択: correlationは報告だけにし、列削除・grid・guard変更を禁止する。

## Inference override設計

- exp264 inference v4と同じraw-test 12候補再生成、40 saved selector、outer別compact 74、clean 273を使う。
- raw train 773 wellsをreferenceにFormationPlaneKNN / DenseANCCImputerをfitし、raw test targetの
  formation列を読まずformation 74を作る。trainと同名wellがあればreference queryからself-excludeする。
- 421列をversion 5の保存済み3 configs × 5 folds = 15 modelへ渡し、平均residualをlast-known TVTへ加える。
- fitting / booster trainingは0。Kaggle private CPU、internet off、run-on-push。外部submit APIは呼ばない。

## 次のアクション

継続status pollingは行わない。ユーザーの完了連絡後にversion 1を1回確認し、必要なoutputだけを取得して
`submission.csv`のsample互換性、行順、重複、NaN/nonfinite、SHAを検証する。competition submitは
別の明示指示がない限り行わない。
