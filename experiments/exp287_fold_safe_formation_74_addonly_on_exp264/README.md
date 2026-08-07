# exp287_fold_safe_formation_74_addonly_on_exp264

## 状態

- ルート: `ml_model`
- 状態: train promotion guard FAIL保持 / inference Kaggle CPU version 1 COMPLETE / submit-check PASS / Public LB 7.530
- CV: `8.136708220359452`
- 親exp264 CV: `8.460811237612477`
- delta: `-0.3241030172530248 ft`
- Public LB / Private LB / Submit ID: `7.530` / - / `54842141`
- 作成日: 2026-07-19
- 完了日: 2026-07-20
- 親実験: `exp264_exp263_candidate_confidence_dual_selector`

## 仮説

exp218監査でfull-train formation reference依存のため除外された74列をouter fold内で作り直すと、
修正版exp264のclean 273 + nested compact 74を維持したまま、hidden-safeなformation情報を
421列のadd-only surfaceとして回収できる可能性がある。

## 変更点

- availability auditの固定74列だけを再生成する。
- outer-train targetは自身をreferenceから除外し、outer-validはouter-train referenceだけを使う。
- 全5 foldのtrain/valid cacheとduplicate/correlation監査をmodel fit前に保存する。
- 保存済みcorrected exp264 347列OOFをcontrolとして再利用し、controlを再学習しない。
- variantは421列の1本、3 LightGBM config × 5 folds = 15 GPU boostersに固定する。
- promotion guard FAILは保持する。2026-07-20のユーザー指示により保存済みmodel inferenceだけを
  overrideし、competition submitは無効のままにする。

## 検証方針

- Fold: exp264 Stage C v6と同じouter 5 group folds
- Group: well
- Metric: RMSE
- Guard: pooled delta `<= -0.02`、4/5 folds改善、near / mid / 1000+ / hidden-like
  delta `<= +0.02`、worst-well `<= +0.25`、+1/+3/+5 ft悪化well数非増加
- Leakage check: self-exclusion、reference well SHA、旧formation列破棄、target formation read禁止、
  fixed audit SHA、fit前feature cache SHA

## 実行入口

- 学習 notebook: `exp287_fold_safe_formation_74_addonly_on_exp264_train.ipynb`
- 推論 notebook: `exp287_fold_safe_formation_74_addonly_on_exp264_inference.ipynb`（inference-only override）
- train kernel: `kentookumura/exp287-foldsafe-form74-addonly-exp264-train` version 5
- inference kernel: `kentookumura/exp287-foldsafe-form74-addonly-exp264-infer` version 1（COMPLETE）
- 実行量: 1 variant × 3 configs × 5 folds = 15/15 GPU boosters、control再学習0
- runtime: `25282.477 sec`（約7時間1分22秒）
- inference runtime: `448.386 sec`、14,151 rows、40 saved selectors + 15 saved TVT models、学習0

## 結果

pooled CVは親exp264の`8.460811`から`8.136708`へ`-0.324103 ft`改善し、5/5 folds、
near / mid / 1000+、hidden-like spatial / typewell-purgedもすべて改善した。

一方、worst well `fb03ae90`は親比`+8.228410 ft`で、+1 / +3 / +5 ft悪化well数も
`135 -> 140` / `39 -> 40` / `14 -> 19`へ増えた。事前固定promotion guardはFAILとなった。

version 1〜4はinput path、親OOF SHA、duplicate projection、formation reference availabilityで
booster fit前に停止したが、version 5は15/15 boostersを完走した。

## 所見

fold-safe formation 74列には強いglobal signalがあり、全fold・全scopeの改善は再現した。ただし改善は
well間で不均一で、事前登録したtail safetyを満たさない。結果に合わせたguard緩和や同一OOFでの
feature/grid救済は行わず、このvariantをtrain-side promotionしない。

## 結論

train-side判定は`train_complete_guard_failed`のままでpromotionしない。ユーザー明示overrideにより
current-test生成と保存済み15 model inferenceを完了し、submission fileはsubmit-check PASSとなった。
ユーザー完了連絡後に確認した提出`ref=54842141`はPublic LB `7.530`で、exp264の`7.562`を`-0.032`
改善したためML routeのLB anchorを更新する。別routeのexp082 ensemble 7.601も`-0.071`で上回るが、
ensemble anchor自体はexp082に維持する。train guard FAILとは分離する。詳細値とSHAは
`result.md`、`metrics.json`、`SESSION_NOTES.md`を正とする。

## 次

即時の救済trainは行わない。exp276のcorrected-parent tail-risk再検証後にだけ、target-freeなformation
tail属性の0-booster attribution readoutを低優先で検討する。実装・実行には別承認が必要。
