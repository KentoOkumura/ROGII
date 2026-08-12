# タスクリスト

## 未着手

- なし。

## 進行中

- なし。

## ブロック中

- Stage 1はStage 0 mechanism FAILのため不適格。
- inference / submissionは無効。branchを閉鎖したため実行しない。

## 完了

- exp209のrate transition式と固定値を確認した。
- exp408のrate under-responseとmomentum単独介入の正負両方の証拠を確認した。
- exp338のglobal`sig_r=0.004`悪化とexp411のtrigger FAILを分離した。
- `mom=0.998 -> 1.0`だけを変更する単一因子仮説を固定した。
- Stage 0 / Stage 1の実行量、gate、禁止事項を固定した。
- `docs/06_reproducibility.md`を読み再現性設計へ反映した。
- steeringをdesign-onlyとして作成した。
- design-only実験scaffoldとconfig / README / SESSION_NOTES / result / metricsを作成した。
- `KAGGLE_DIRECTION.md`の未着手バックログへP3として追加した。
- `experiment_summary.md`へdesign-only実験を反映した。
- strict experiment validationとtemplate validationを通した。
- ユーザーの追加依頼により実装と正規Notebook採用の承認を得た。
- compact self-contained Jupytext train候補を実装した。
- fail-closed inference placeholderと専用contract testを実装した。
- exp411 fixed32 manifestとexp408 episode ledgerをSHA検証contractへ接続した。
- exp209 untreated small-trellis parity、`mom=1.0` transition mean、
  truth-late境界、inference禁止をtestした。
- exp411 / exp412 compactと章立て・行数を比較し、同じ9章構成で
  exp411 2255行、exp412 2333行、exp424 2237行と確認した。
- compact / 正規train・inference NotebookをJupytextで生成・採用した。
- py_compile、Ruff F821/F401、専用pytest 10件、Jupytext `--test`、
  strict experiment validationを通した。
- Stage 0はbaseline 32 + treatment 32 = 64 HMM well-runs、
  model / booster / PF / Beam / GPU 0と再確認した。
- ユーザーから事前登録済みStage 0のKaggle実行承認を得た。
- strict packageをcanonical kernel idへpushした。
- Kaggle private CPU Version 1（id `128924158`）で64 HMM well-runsを完走した。
- technical gate 13 / 13 PASS、mechanism gate 3 / 7 PASSと判定した。
- runtime `2,077.533832秒`、peak RSS `1.030926 GB`、artifact SHAを記録した。
- Kaggle output実体を一時取得し、metrics / summary / prediction /
  rate readout / well / episode / input manifest SHAを照合した。
- `stage0_fail_closed`としてStage 1、inference、submissionなしでbranchを閉じた。
