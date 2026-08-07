# タスクリスト

## 未着手

- Stage 0完了後にgateとSHAを記録する。

## 進行中

- 正規Notebook採用、Kaggle package / push / run。

## ブロック中

- parent controlを含むStage 1 Kaggle実行は別承認待ち。

## 完了

- exp408のbeta-filter disagreement / backward reversal証拠を確認した。
- two-pass triggerとrate-only transition変更を一意に固定した。
- Stage 0 / Stage 1の対象、実行量、gate、禁止事項を固定した。
- exp411に対する優先順位とclose条件を固定した。
- `docs/06_reproducibility.md`を読み再現性設計へ反映した。
- steering、実験scaffold、backlogをdesign-onlyとして作成した。
- exp411 Stage 0 fail-closeにより先行条件成立を確認した。
- ユーザーの2026-07-28実装指示を確認した。
- compact self-contained two-pass Jupytext train候補を実装した。
- fail-closed inference候補と専用contract testsを実装した。
- fixed32 cause-stratified sample manifestを生成しSHAを固定した。
- dedicated test 12件、Jupytext、py_compile、Ruff F821、strict validationを通した。
- ユーザーの2026-07-28実行指示によりStage 0 64 HMM well-runsの実行承認を得た。
