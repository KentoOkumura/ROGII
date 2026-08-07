# タスクリスト

## TODO

- 正規inference notebookへの採用前にユーザー承認を得る。
- Kaggle push前にvariant 1、model/config/fold/booster/PF/Beam/GPU各0を`SESSION_NOTES.md`へ再記録する。
- Kaggle package metadataとbootstrap内config、weight、input SHAをreadbackする。
- Kaggle inference実行、output取得、submit-check、外部提出はそれぞれ別承認で行う。
- output取得後にprediction content SHA、submission SHA、kernel versionを記録する。

## 進行中

- なし

## ブロック中

- 正規notebook採用、Kaggle package/run、output取得、提出は未承認。

## 完了

- 2026-08-04: ユーザー指定の最終提出第1枠として、係数、入力、禁止処理、reference-only判定を設計確定。
- 2026-08-04: 再現性設計を記入。
- 2026-08-04: experiment scaffold、config、ensemble/output contract、README/result/metrics/SESSION_NOTESを作成。
- 2026-08-04: strict experiment validation、template validation、summary/backlog更新を完了。
- 2026-08-04: exp497 Stage I version 4のmodel/prediction/artifact SHAを固定。
- 2026-08-04: prediction-only compact self-contained inference source/notebookを実装。
- 2026-08-04: exp413中間submission隔離、component別visible parity、ID/order/finite/SHA、
  float64 fixed formula、truth-free差分、reproducibility manifestを実装。
- 2026-08-04: dedicated test 6件、exp497 dependency test 30件、Jupytext、py_compile、Ruff、
  strict experiment/template validationをPASS。
