# タスクリスト

## 未着手

- promotion PASS時だけ、同じexp402内のinference実装を別途相談する。

## 進行中

- Stage 1 GPU train version 4の完了後、保存OOFと固定promotion gateを確認する。

## ブロック中

- inferenceはpromotion PASSと別承認が必要。
- submissionはsubmit-check後の別承認が必要。

## 完了

- [x] 2026-07-26: exp402を採番し、route、親、clean tail controlを固定。
- [x] 2026-07-26: 固定8候補、GRWR 5列、float32演算、列順を固定。
- [x] 2026-07-26: outer-train / outer-valid / current-testのformation境界を固定。
- [x] 2026-07-26: 0-booster preflightとpromotion gateを固定。
- [x] 2026-07-26: 条件付き学習量を
  1 variant / 3 configs / 5 folds / 15 GPU boosters / control 0に固定。
- [x] 2026-07-26: 再現性、禁止事項、実装・実行の承認境界を固定。
- [x] 2026-07-26: ユーザーの実装指示を受け、親exp287の9 markdown
  cell / 362行に対して、11 numbered chapters / 2,263行の別名
  compact self-contained train候補を実装。
- [x] 2026-07-26: 旧exp218 generatorを丸ごと呼ばず、必要な
  DWT/FFT/NCC 3成分だけを同じ固定式で再生成する実装を追加。
- [x] 2026-07-26: matching exp287 outer-role 10 partition、GRWR-5
  schema/content SHA、raw current-test regeneration、target formation read 0を実装。
- [x] 2026-07-26: fail-closed inference候補と専用test 8件を追加。
- [x] 2026-07-26: 候補`.ipynb`へJupytext変換し、pycompile、Ruff、
  Jupytext round-trip、strict `validate-exp`、`8 passed`を確認。
- [x] 2026-07-26: ユーザーの「実行してください」により、正規train
  Notebook採用と0-booster private CPU package/push/runを承認済みとして固定。
- [x] 2026-07-26: version 1の最終statusを
  `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`と確認。保持ログは15.354秒の
  Stage 0本処理直前まで、tracebackなし、Kaggle公開output 0件。
  runtime上限が最有力だがAPIで手動cancelと区別できないため、原因は推定として記録。
- [x] 2026-07-26: ユーザーの「設計変更と再実行を進めてください」により、
  train roles / current test / aggregateの3-run分割とprivate CPU retryを承認済み。
- [x] 2026-07-26: 3 wrapper、phase manifest、source/config SHA一致、
  upstream file SHA再検証を実装。Jupytext、pycompile、Ruff、strict validation、
  専用test `10 passed`。
- [x] 2026-07-26: 0A/0B/0C packageを同一config SHA
  `98dd377e...6176`、source SHA`665f41ad...fb20`で凍結。
- [x] 2026-07-26: 0A version 1をpush。private、CPU、internet off、
  input 3件、id_no `128687498`をpullで確認。
- [x] 2026-07-28: train source、current-test、fold 0–4のupstream 7 runが
  PASSし、aggregate version 1をpush。
- [x] 2026-07-28: aggregate version 1がfold 4 path解決で失敗したことを
  tracebackで確認。同名sentinel fallbackが5候補を返すcode defectと特定。
- [x] 2026-07-28: aggregate wrapperへfold 4 v2の明示runtime aliasを追加。
- [x] 2026-07-28: 同名sentinelを持つ5 fold inputの回帰testを追加し、
  専用test `11 passed`。
- [x] 2026-07-28: config / compact implementation source SHA不変を確認し、
  aggregate Notebook/packageを再生成。Jupytext、pycompile、Ruff、
  strict validationをPASS。
- [x] 2026-07-28: 同じcanonical aggregate slugへversion 2をpush。
  id_no `128831850`、private CPU、internet off、status RUNNINGを確認。
- [x] 2026-07-28: aggregate version 2の終端status
  `KernelWorkerStatus.COMPLETE`を確認。
- [x] 2026-07-28: preflight `18 / 18` checks、10 outer-role partition、
  current-test 14,151 rows / 3 wells、historical GRWR load・target formation read・
  model / booster / prediction / submission各0を確認し、Stage 0 technical gateをPASS。
- [x] 2026-07-28: partition / preflight / reproducibility manifestの
  file SHAを取得して記録。
- [x] 2026-07-28: Stage 1 trainの明示承認を受け、
  1 variant / 3 configs / 5 folds / 15 T4 boosters / control再学習0を
  `SESSION_NOTES.md`へpush前に固定。
- [x] 2026-07-28: 426列のStage 1 compact self-contained Jupytext候補、
  fail-closed inference状態更新、専用testを実装し、正規train Notebookへ採用。
- [x] 2026-07-28: Stage 1 version 2をpush。aggregate input mount pathを
  発見できず、10.6秒・0 boosterでtechnical failureしたことを確認。
- [x] 2026-07-28: required manifest名と固定file SHAによるartifact root探索、
  物理T4 runtime guard、回帰testを追加。専用test `13 passed`、
  Jupytext、pycompile、Ruff、strict validation、package SHA監査をPASS。
- [x] 2026-07-28: 同じcanonical kernelへStage 1 version 3をpush。
  id_no `128627922`、T4 requested、private、internet off、10 input、
  status `KernelWorkerStatus.RUNNING`を確認。
- [x] 2026-07-28: Stage 1 version 3が物理T4 ×2とSHA-qualified rootを確認後、
  exp145 input不足で227.3秒・0 boosterのtechnical failureとなったことを確認。
- [x] 2026-07-28: `exp145-train`を追加し、固定11 inputと必要3ファイルの
  fail-fast guard、回帰testを追加。Jupytext、pycompile、Ruff、strict validation、
  専用test `13 passed`、package SHA監査をPASS。
- [x] 2026-07-28: 同じcanonical kernelへStage 1 version 4をpush。
  id_no `128627922`、T4 requested、private、internet off、11 input、
  status `KernelWorkerStatus.RUNNING`を確認。
