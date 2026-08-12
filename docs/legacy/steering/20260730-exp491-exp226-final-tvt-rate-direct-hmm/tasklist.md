# タスクリスト

## 現在の状態

`stage0_fail_closed`

## 完了

- [x] HMM を先行し、PF は exp491 の結果レビュー後までブロックする方針を確定した。
- [x] exp437 を構造上の親、exp226 最終 `tvt_pred` を schedule source として確定した。
- [x] 変更する科学的変数を「exp226 最終 `tvt_pred` の行間増分による HMM 遷移中心」だけに固定した。
- [x] rate state、smoothing、clipping、blend、selector、PF を exp491 の範囲外とした。
- [x] Stage 0 固定 32 well と、別承認が必要な Stage 1 full OOF の gate を事前登録した。
- [x] 入力 content SHA、列 allowlist、truth join timing、raw-GR evidence reuse risk を記録した。
- [x] 実験 scaffold、config、README、SESSION_NOTES、result、metrics の設計状態を作成した。
- [x] HMM 現行案と条件付き PF 後続案をバックログへ記録した。
- [x] exp491 を experiment summary へ設計段階として記録した。

## 実装完了

- [x] Jupytext percent 形式の別名 train source を作る。
- [x] exp437 の HMM を移植し、schedule source だけを exp226 最終 `tvt_pred` に交換する。
- [x] exp226 OOF の strict `usecols`、logical SHA、truth-freeze guard を実装する。
- [x] first-difference parity、rate identity、normalization、coverage、実行ロックのテストを作る。
- [x] notebook 変換、Jupytext round-trip、構文、F821、契約テストを行う。
- [x] Stage 0 の variant 数 1、HMM well-run 32、booster 0 を `SESSION_NOTES.md` で再確認する。
- [x] 親 exp437 と章構成・行数を比較し、9章対9章、1948行対1714行、
  `__file__` 参照0を確認する。
- [x] 2026-07-31の実行承認を記録し、compact実装を正規train notebookへ採用する。
- [x] fixed32 manifestとpersistent episode assetをbootstrapへ固定し、
  canonical private CPU packageを生成・検証する。
- [x] version 1で32/32 HMM wellsの完了後にgzip終端未書き込みの
  `EOFError`を確認し、入れ子context managerとreadback回帰テストで修正する。
- [x] 同一科学契約のKaggle private CPU version 2を完了する。
- [x] technical gate全件PASS、mechanism gate 1/7 PASS・6/7 FAILを記録する。
- [x] `result.md`、`metrics.json`、`SESSION_NOTES.md`、README、方向性を更新する。

## ブロック中

- [x] Stage 0 実行: version 2（id_no `129213586`）COMPLETE、fail-closed。
- [x] Stage 1 full OOF: Stage 0 mechanism gate不合格のためブロック。
- [x] inference: Stage 1へ進まないためブロック。
- [x] PF 後続実験: HMM不合格時のPF救済禁止に従い閉鎖。
- [x] submission: inferenceへ進まないためブロック。

## 停止条件

- Stage 0 の gate が 1 つでも不合格なら exp491 を終了する。
- Stage 1 の promotion gate が 1 つでも不合格なら inference へ進まない。
- 同じ exp 内のパラメータ追加、別 rate 補正、PF 化で negative result を救済しない。
