# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- なし

## 完了

- steering要件、設計、再現性方針を記載した。
- active variant 1、LightGBM config 0、fold 0、booster 0、control再学習なしを確定した。
- raw train/test GR missing-run readoutを実装した。
- exp221 control cache / exp148 OOF source resolverとstrict ID/SHA guardを実装した。
- exact HMMのGR mask-only変更とsynthetic contract testを実装した。
- overall、missing-run、post-gap、distance、hidden-like、worst-well、finite coverage、divergence readoutを実装した。
- Jupytext trainとno-inference notebookを通常 `.ipynb` へ変換した。
- strict validation、py_compile、Ruff、Jupytext testを通した。
- Kaggle CPU train packageを作成し、metadataとbootstrap内configの整合を確認した。
- canonical Kaggle CPU train version 1（id_no `127064272`）を完走した。
- kernel status/logsと必要な小規模artifactだけを取得し、overall、missing-run、post-gap、distance、hidden-like、by-well、finite coverage、divergenceを監査した。
- input/output raw・decompressed SHA、runtime、variant/config/fold/booster数を`SESSION_NOTES.md`と`metrics.json`へ記録した。
- 一律maskはtiny RMSE gain、MAE悪化、hidden-like spatial悪化、worst-well +2.576981 ftのため不採用とし、run-length gate / inference / submitへ進まずbranchを閉じた。
