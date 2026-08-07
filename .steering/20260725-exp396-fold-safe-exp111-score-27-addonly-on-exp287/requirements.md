# 要件

## 依頼

exp111 score系27列をfold-safeに作り直し、exp287の421特徴へadd-onlyする実験を設計する。
初回turnでは設計文書とscaffoldだけを作成した。2026-07-25の後続指示
`exp396を実装してください` により、Stage Aのcompact self-contained候補、fail-closed
inference候補、専用testまでを実装する。その後の明示承認により正規train notebook採用、
Kaggle package/push、0-booster preflightまでを実行する。さらに2026-07-25の後続指示で
固定40 CPU booster学習を承認する。その完了後のユーザー指示 `実行してください` により、
Stage B実装と固定15 GPU boosterのKaggle T4実行を承認する。推論、提出は含めない。

## 制約

- 対象実験は `exp396_fold_safe_exp111_score_27_addonly_on_exp287`、Routeは `ml_model`、
  親実験は `exp287_fold_safe_formation_74_addonly_on_exp264` に固定する。
- 親の `clean 273 + nested compact 74 + fold-safe formation 74 = 421` 特徴、outer 5-fold、
  target、LightGBM 3 config、early stopping、seed、評価行を固定し、追加する変数は27列だけとする。
- exp111保存済みfold0 classifier / expected-error regressorをtrain全体へ適用した27列は使わない。
  exp111の候補・48入力特徴・2目的・LightGBM設定は仕様参照に限り、モデル重みは再利用しない。
- 各downstream outer foldの内部を4 GroupKFoldに分け、outer-train行はinner OOF、
  outer-valid行はそのouter foldの4 inner model平均でscoreを生成する。
- 2目的は `candidate_within10_binary` と `candidate_abs_error`、候補は
  `pf_ancc / beam_mean / likpf_mean / sc_ens / hyb` の5個に固定する。
- scorerの各fitでouter-valid wellを参照しない。outer-trainの各行も、その行のwellを学習に含む
  scorerからscoreを受け取らない。
- candidate-long subsampleはstable SHA256由来のlocal RNGだけで行い、global RNGやthread順序に
  sample集合を依存させない。
- scorer入力48列のimputation medianは各inner-trainだけでfitしてモデルと一緒に保存し、
  inner-valid、outer-valid、current-testへ適用する。batch medianやfull-train medianは禁止する。
- exp287保存済み10 fold-role formation cacheをSHA固定して再利用し、formation 74を再生成しない。
- exp264で無効判定された依存GRWR 6列は追加しない。score加重TVT、hard top1、direct blend、
  sample weight、feature subset、parameter grid、control再学習も含めない。
- Stage Aは `5 outer × 4 inner × 2 objectives = 40` CPU boosters。Stage BはStage Aの全gateを
  PASSした後だけ候補化し、`1 variant × 3 configs × 5 folds = 15` GPU boosters、
  control再学習0とする。
- Stage A実装、0-booster preflight、固定40 CPU booster実行は承認・完了済み。
  Stage B実装・GPU実行も15 boostersとcontrol再学習0を再提示して明示承認済み。
  inference、submissionは自動承認しない。
- Kaggle Notebook実行を正とし、internet offで成立させる。

## 受け入れ基準

- 27列の固定allowlist、列順、由来が文書とconfigで一致する。
- nested score生成のouter/inner fit境界とcurrent-test生成契約が一意に定義されている。
- 40 scorer model、model固有48列median、feature schema、outer/inner well集合、logical feature
  contentのSHA記録口が定義されている。
- Stage Aのleakage/coverage/score/resource gateと、Stage Bのglobal/fold/scope/by-well/tail
  promotion gateが実行前に固定されている。
- Stage Bの最終surfaceが421 + 27 = 448列、active variant 1、3 configs、5 folds、
  15 GPU boosters、control再学習0である。
- stochastic component、stable seed、CPU/GPU deterministic設定、artifact SHA方針が
  `docs/06_reproducibility.md` に沿って明記されている。
- `KAGGLE_DIRECTION.md` と `experiment_summary.md` にStage A全gate PASS、Stage B 15/15完了、
  固定promotion gate FAILによるbranch閉鎖が記録されている。
- 明示承認後に正規train notebookを採用し、別名のJupytext候補、変換notebook、専用test、
  設定・文書検証が通る。Stage A成果物はSHA固定して保存する。

## 次のアクション

Stage Bは固定15/15 GPU boostersを完走したがpromotion gateをFAILしたため、
exp287をtrain-side parent anchorに維持し、inference / submissionへ進まない。
