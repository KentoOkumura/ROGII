# タスクリスト

## 目的

exp416の局所回復と全体破壊を、保存生成物だけで事前固定したwell regimeへ帰属する。

## 変更点

exp416を再実行せず、保存済みdiagnostic / prediction / outcomeをtruth-late順に読み、
outer-4-fold ECDFの2軸regimeと固定row scopeだけを原因帰属に使う。

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- なし。

## 完了

- `exp422_roughening_x10_failure_regime_attribution_readout`を採番した。
- steering requirements / design / tasklistを作成した。
- route、親、入力kernel version / manifest SHA、0-PF / 0-model実行量を固定した。
- target-free読込順、6 raw diagnostics、2 score、fold-safe経験分布順位、
  median 2軸、1 target cellを固定した。
- fixed位置readout、4096回置換、fold再現性、episode supportを含むAND gateを固定した。
- adaptive roughening、threshold探索、same-OOF rescue、inference、submissionを対象外にした。
- design-only experiment scaffoldと記録ファイルを作成した。
- ユーザーから実装の明示承認を得た。
- Jupytext percent形式のcompact self-contained train候補と`.ipynb`を作成した。
- source manifest / terminal contract、truth-late ledger、fold-safe ECDF / median、
  fixed cell / row scope、4096回fold内置換、metrics / gateを実装した。
- 専用contract tests 9件、Jupytext test、`py_compile`、Ruff F821、
  strict experiment validationをPASSした。
- 正規train / inference Notebookはplaceholderのまま保持し、Kaggle package / push /
  audit runは行っていない。
- ユーザーから正規train Notebook採用、Kaggle package / push / CPU audit runの
  明示承認を得た。inference / submissionは未承認のまま保持した。
- 正規train Notebookを採用し、canonical private CPU kernelをpackageした。
- version 1の親logical SHA列契約不一致を診断し、科学設定を変えず修正した。
- version 2を完走し、3,783,989 rows / 773 wells、4096回fold内置換を確認した。
- technical gateをPASSした。
- recovery-pressure、damage-exposure、固定target cellのscientific gateをFAILした。
- result / metrics / SESSION_NOTES / experiment summary / directionを更新した。
- exp416のterminal FAILを維持し、attribution branchを救済なしで終了した。
