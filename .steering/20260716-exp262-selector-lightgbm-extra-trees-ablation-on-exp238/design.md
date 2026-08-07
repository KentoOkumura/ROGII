# 設計

## アプローチ

exp238 selector train v4の候補面とstrict nested stackingを再構築し、各inner modelの
LightGBM paramsへ `extra_trees=True` だけを追加する。保存済みexp238 nested OOF scoreを
同じouter-valid rowへ整列してfrozen controlとし、新旧scoreを同じ評価関数へ通す。
selector出力は候補誤差scoreとしてのみ評価し、exp218 downstream TVT LightGBMへは入力しない。

historical controlと新variantについて、候補誤差のcalibration/ranking、score相関、row-wise
top1、exp237で固定済みのViterbi rule、距離bucket、exp115 hidden-like、fold、well別を保存する。
guardは新variantがhistorical controlに対して非悪化であることと、exp238既知のworst-well
回帰を拡大しないことを要求する。

## 実験範囲

- 対象実験: `exp262_selector_lightgbm_extra_trees_ablation_on_exp238`。
- Route: `ml_model`。
- 親実験: `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`。
- 変更する変数: selector LightGBMの `extra_trees: true` だけ。
- 固定する変数: train v4候補11本、context 184列、candidate-long追加3列、outer/inner split、行上限、seed、objective、early stopping、Viterbi rule、fallback、評価bucket。
- 実行量: active variant 1 × selector config 1 × outer 5 × inner 4 = 20 CPU boosters。control/downstream再学習0。

## 再現性設計

- seed policy: exp238 seed 42、outer/inner indexから作るmodel seed、bounded sampling seedを完全固定する。
- stochastic 処理の有無: bounded row samplingとLightGBM extra-trees threshold選択がstochastic。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。保存・再構築されたcandidate valueを固定入力として使う。
- 並列処理と乱数の関係: samplingはモデル単位のlocal `np.random.default_rng(seed)`、LightGBMは固定random stateとCPU thread設定を使う。
- CPU/GPU runtime と deterministic flags: CPU、固定`n_jobs`。GPUは使わない。rerun一致確認前はdeterministic anchorと呼ばない。
- train cache / test feature regeneration の SHA 記録方針: 入力cache、context schema、historical nested scoreのdecompressed SHAを記録する。raw-test regenerationは初回範囲外。
- model manifest / prediction / submission SHA 記録方針: 20 model SHA、model manifest SHA、新nested score decompressed SHA、comparison table SHAを記録する。submissionは生成しない。
- Kaggle package bootstrap 確認方針: prepare後にsupport files、config、kernel sources、internet/GPU metadataの一致を確認してからpush可否を別途判断する。

## リスク

- リークリスク: outer-valid wellがinner train/validへ入らないassertを維持し、historical scoreもrole/fold/row/candidate契約を検証する。
- CV/LB 不一致リスク: selector単体改善がdownstream exp218やPublic LBへ転移するとは限らないため、初回はselector-only guardに限定する。
- ランタイム/メモリリスク: 3.78M rows × 11 candidatesのfull predictionが重い。学習/validationはbase-row上限、推論はchunk処理、historical scoreはfold単位読み込みを使う。
- 再現性リスク: LightGBMとthread schedulingによりbitwise差が出る可能性がある。seed、thread数、input/model/prediction SHAを保存し、必要なら同一kernel rerunで確認する。

## 次のアクション

静的検証とKaggle package監査後、20 CPU boosters、control/downstream再学習0のscopeを
ユーザーへ再提示する。明示承認を得た場合だけ同じexp262のcanonical train kernelをpushする。
