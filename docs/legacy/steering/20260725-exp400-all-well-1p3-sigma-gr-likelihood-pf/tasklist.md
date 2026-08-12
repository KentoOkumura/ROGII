# タスクリスト

## Design-only完了

- [x] `exp400_all_well_1p3_sigma_gr_likelihood_pf` を採番した。
- [x] Routeを `pf_beam`、scientific parentをexp072 deterministic v2に固定した。
- [x] discussionの `lik_pf` 1行をlocal deterministic PFへ移す実験であり、
  公開Notebook全体のscore再現ではないことを明記した。
- [x] `gs_candidate = 1.3 * clip(gs_raw, 10, 60)`、再clipなし、全773 wellsを固定した。
- [x] 500 particles / 128 seeds / scales 3,5,8,12 / exp072 dynamicsと
  stable SHA256 per-well seedを固定した。
- [x] primaryを `likpf_mean`、scale outputsを非選択secondary diagnosticに固定した。
- [x] saved exp072 controlとsaved exp209 fixed 50:50 guardをload-onlyに固定した。
- [x] 1 variant / 773 PF wells / 98,944 seed-well trajectories /
  49,472,000 particle starts / 5 reporting folds / booster 0 /
  parent control再実行0を固定した。
- [x] truth-late freeze、technical gate、scientific promotion gate、
  fail-close条件、禁止事項を固定した。
- [x] backlog、steering、experiment scaffold、experiment summaryを
  design-only状態で登録した。

## Implementation-only完了

- [x] implementation-only承認を得る。
- [x] Jupytext percent形式のcompact self-contained train候補を別名で作る。
- [x] inferenceをsubmission非生成のfail-closed compact候補にした。
- [x] exp072 x1.0 fixture parity、1.3適用順、seed、truth-late、
  execution count、gateを固定する専用testを作る。
- [x] Jupytext round-trip、py_compile、Ruff、専用testを通す。
- [x] strict experiment validationを通す。

## Kaggle実行完了

- [x] 正規train Notebook採用の承認を得る。
- [x] Kaggle private CPU package / push / runの承認を得る。
- [x] push前に1 variant / 773 PF wells / 98,944 seed-well /
  49,472,000 particle starts / model・booster・control再実行0を再確認する。
- [x] Kaggle上でcandidateだけを生成し、technical gateとartifact SHAを監査する。
- [x] truth-lateでprimary promotion gateを評価し、FAILをterminal記録する。
- [x] FAILのためinference設計・実装へ進めないことを記録する。
- [x] submit-check / Kaggle submissionを実施しないことを記録する。

## Terminal result

- [x] Kaggle private CPU version 1 / id_no `128585102`を完走した。
- [x] technical gate PASS、scientific gate FAILを確認した。
- [x] candidate/control RMSE
  `12.221810980460939 / 11.594894395642696`、
  improvement `-0.6269165848182432 ft`を記録した。
- [x] fixed HMM 50:50、fold、stress scope、by-well tailがFAILしたことを記録した。
- [x] 小型outputのmanifest SHAを実ファイルと照合した。
- [x] decision
  `all_well_likelihood_pf_gs_x1p3_failed_close_without_rescue`
  としてbranchを閉じた。

## 現在禁止

- version 2または別slugへの再push
- parent PF / HMM / Beam / modelの再実行
- inference / submission
- multiplier / clip / particle / seed / scale / resampling / blend /
  selector / adaptive well gateによるsame-OOF救済
