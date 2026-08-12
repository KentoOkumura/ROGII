# 要件

## 依頼

exp264 availability auditで無効とされたGRWR 6列のうち、formation依存の
5列だけをexp287のfold-safe formation生成物から再計算し、exp287へ
add-onlyする実験を設計する。backlog、steering、実験ディレクトリと
設計を確定する。2026-07-26の追加依頼「exp402を実装してください」により、
別名Jupytext候補、専用test、0-booster preflight実装までを追加する。
その後の「実行してください」で正規Notebook採用とStage 0 version 1を実行したが、
`CANCEL_ACKNOWLEDGED`、公開output 0件で未完了となった。追加依頼
「設計変更と再実行を進めてください」により、Stage 0をtrain roles、
current test、aggregateへ分割実装し、private CPUで再実行する。
aggregate version 1のfold 4 input path解決失敗後の追加依頼
「修正と再実行してください」により、upstream execution identityを維持した
wrapper-only path alias修正とaggregate version 2再実行を行う。
15 GPU booster学習、推論、提出は行わない。

## 制約

- Route: `ml_model`
- 親実験:
  `exp287_fold_safe_formation_74_addonly_on_exp264`
- clean tail control:
  `exp264_exp263_candidate_confidence_dual_selector`
- 再現性: `docs/06_reproducibility.md`に従い、fold-role feature、
  current-test再生成、GPU、Kaggle bootstrap、SHA記録を設計に明記する。
- exp287のclean 273 + nested compact 74 + fold-safe formation 74 =
  421列、outer folds、target、LightGBM 3 configs、非加重評価を固定する。
- 旧exp218のGRWR 5列値は使わず、各outer foldのexp287 formation roleから
  決定論的に再計算する。
- 候補TVTは固定8候補とし、候補subset、spread式、dtype、interactionを探索しない。
- 追加するのはformation依存GRWR 5列だけとする。
- exp396 score-27、`grwr_ll_entropy_x_dwt_energy_ratio_w065`、
  sample weight、error-segment weight、hard gate、direct TVT correctionを含めない。
- 0-booster preflightはmodel、booster、prediction、submissionを生成しない。
- 分割後も式、候補、fold、dtype、row identity、SHA定義を変更しない。
- Stage 0Aはtrain source componentsと10 outer-role partitionsだけを生成する。
- Stage 0Bはcurrent-test 3 wellsのraw PF/Beam replayだけを生成する。
- Stage 0CはA/Bのimmutable output file SHAとlogical-content ledgerを統合し、
  feature/PF生成を再実行しない。
- 学習実装とKaggle trainは別承認とする。
- 将来の学習は1 variant / 3 LightGBM configs / 5 folds /
  15 GPU boosters、親control再学習0に固定する。
- promotion gateを一つでもFAILした場合は同一OOF救済なしで閉じ、
  inference、submissionへ進めない。

## 受け入れ基準

- `docs/legacy/steering/20260726-exp402-fold-safe-grwr-5-addonly-on-exp287/`に
  仮説、5列の式、fold-safe生成、preflight、学習量、promotion gate、
  禁止事項が固定されている。
- `experiments/exp402_fold_safe_grwr_5_addonly_on_exp287/`に
  design-onlyの`config.yaml`、`README.md`、`SESSION_NOTES.md`、
  `result.md`、`metrics.json`とtemplate placeholder notebookがある。
- `KAGGLE_DIRECTION.md`の未着手backlogと`experiment_summary.md`に
  exp402が記録されている。
- 親421列、追加5列、最終426列が明記されている。
- 候補8列、標準偏差`ddof=0`、range、3 interaction、float32列順が固定されている。
- outer-train self-exclusion、outer-valid outer-train-only、
  current-test all-train reference、target formation read 0が明記されている。
- 0-booster preflightと15 GPU booster学習の承認境界が分離されている。
- 親exp287とclean exp264の保存OOF SHAを固定し、control再学習0を明記している。
- gzip生成物はraw gzip SHAではなくdecompressed content SHAを主証拠とする。
- 別名Jupytext train/inference候補と専用testがあり、正規Notebookは
  placeholder SHAのまま未採用である。
- Kaggle package/run、model、booster、prediction、submissionがすべて0である。
- Stage 0 current-test再生成量が3 test wells、PF ANCC 3、PF Z 3、
  Beam path 21、likelihood-PF 3 well-runs / 384 seed-well trajectories /
  192,000 particle startsと明記されている。
- Stage 0A/0B/0Cが同じimplementation source SHA、config SHA、
  scientific contract SHAを持ち、Stage 0CがA/Bのfile SHAを再検証する。
- version 1を上書き再実行せず、分割runは役割が分かるprivate canonical slugで
  実行され、各kernel id/version/runtimeを記録する。
