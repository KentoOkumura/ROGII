# タスクリスト

## TODO

- なし

## ブロック中

- なし

## 完了

- 再現性設計を `design.md` に記入した。
- exp245は41特徴削除ablationとして残し、exp238の保存済みモデルを再利用する方針を固定した。
- exp238内にraw-test copcf parity監査notebookを実装した。
- Jupytext、py_compile、ruff、strict experiment validationを通した。
- CPU、internet off、saved selector 20適用、selector/final学習0、submissionなしのKaggle packageを生成した。
- bootstrap内configとexp218 replay、exp226、exp237 source/configのSHAを確認した。
- ユーザー承認後にCPU / internet off / run_on_push=trueでKaggle v1を実行した。
- 184 context、41 copcf、exp226診断4列、missing列0、全nonfinite値0を確認した。
- 保存済み20 modelのouter/inner完全被覆、5 score面各14,151行、生成物SHAを実ファイルで検証した。
- parity受け入れ基準をすべて満たしたため、正規current-test inferenceへ採用すると判断した。
- parity通過済みgeneratorを保存済み15 final LightGBMのhidden-safe inferenceへ接続した。
- 別名Jupytext notebook、T4/internet off/run_on_push=false package、bootstrap configを静的検証した。
- ユーザー承認後にT4 / internet off / run_on_push=trueでparity-integrated final inference v1を実行した。
