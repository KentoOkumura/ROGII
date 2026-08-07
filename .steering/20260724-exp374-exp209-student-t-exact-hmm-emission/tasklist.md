# タスクリスト

## 未着手

- なし。

## 進行中

- なし。

## ブロック中

- なし。固定gate FAILによりbranchをterminal closeした。

## 完了

- 2026-07-24: exp342がexp281 residual-offset HMMを親としていたことを確認した。
- 2026-07-24: ユーザー意図をexp209 absolute-TV​T exact HMMのGaussian emission
  単独置換として確定した。
- 2026-07-24: exp372使用済み、exp373 steering予約済みを確認し、exp374を採番した。
- 2026-07-24: `docs/06_reproducibility.md`を確認した。
- 2026-07-24: direct full-HMM、fixed`df=4`、保存control、truth-late join、
  scientific/tail gate、no-rescue、実行量をsteeringへ固定した。
- 2026-07-24: design-only実験scaffoldを作成し、implementation/run/inference/
  submission flagをすべて`false`へ固定した。
- 2026-07-24: `KAGGLE_DIRECTION.md`の判断メモと未着手バックログ、
  `experiment_summary.md`へ登録した。
- 2026-07-24: YAML/JSON parse、exp strict validation、project strict validation、
  design contract audit、文書reviewをPASSした。
- 2026-07-24: compact self-contained train候補へ固定`df=4` Student-t emission、
  exp209 observation/state parity、prediction freeze、late truth/control join、
  fixed scope/gateを実装した。
- 2026-07-24: fail-closed inference候補と専用contract testを実装し、`9 passed`。
- 2026-07-24: 親exp209 Student-t mode / exact kernel parity、py_compile、Ruff、
  Jupytext変換・round-trip、notebook-safe pathを確認した。
- 2026-07-24: exp/project strict validation、`make validate-exp`、実験文書reviewを
  PASSし、実装済み案を`KAGGLE_DIRECTION.md`の未着手バックログから削除した。
- 2026-07-24: ユーザーが正規train Notebook採用とKaggle package/push/runを承認した。
- 2026-07-24: `1 variant / 773 HMM / model・fold・booster・control rerun各0`
  を再確認し、3つのKaggle kernel sourceに必要成果物が存在することを確認した。
- 2026-07-24: compact self-contained train候補を正規train Notebookへ採用した。
- 2026-07-24: Kaggle private CPU version 1、id_no `128436182`を
  1 variant / 773 HMM runs / control再実行0で完了した。
- 2026-07-24: technical gate PASS、direct`+0.217809 ft`、4/5 folds改善を確認した。
- 2026-07-24: by-well p95`+0.982661 ft`、worst`+35.015963 ft`でtail gateを
  FAILし、`student_t_exp209_failed_close_without_rescue`として閉じた。
