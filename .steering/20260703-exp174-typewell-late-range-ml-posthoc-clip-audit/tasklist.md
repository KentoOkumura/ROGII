# タスクリスト

## 未実行

- Kaggle train notebook を prepare / push する場合は、config と bootstrap の整合、exp148 train source の有無を確認する。
- Kaggle train 完了後、logs / cell output から baseline、best grid、changed rows、bucket guard、生成物 path、SHA を `SESSION_NOTES.md` / `result.md` に記録する。
- positive の場合だけ、raw-test parity、front-half exception well、worst-well regression を追加確認する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering requirements / design を作成した。
- 実験 scaffold を作成した。
- `config.yaml`、監査 module、train / inference percent notebook source を実装した。
- train / inference ipynb を Jupytext で生成した。
- `py_compile`、`ruff --select F821`、Jupytext round-trip、ipynb JSON 検証、`make validate-exp` を通した。
- Kaggle train package を strict prepare し、kernel source に exp148 / exp092 / exp073 を含めた。
