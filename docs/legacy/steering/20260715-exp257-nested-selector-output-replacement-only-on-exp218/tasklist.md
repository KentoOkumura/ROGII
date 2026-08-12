# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- なし。CV guard不通過を確定し、不採用で終了した。

## 完了

- exp218の既存`ll_*` 54列をselector出力29列と入力診断25列へ分類した。
- `nsel_*`を追加しない380列replacement-only契約を確定した。
- 再現性設計を`design.md`に記入した。
- train notebook、replacement engine、config、実験記録を実装した。
- 実exp218 schemaと合成11候補によるcontract test、Jupytext、py_compile、ruff、strict experiment validationを通した。
- `run_on_push=false`のKaggle T4 train packageを作成した。
- ユーザーのGPU実行承認を得た。
- 1 variant / 3 configs / 5 folds / 15 boosters / control 0を再確認した。
- metadataとbootstrap内configを一致させ、canonical kernel v1へpushした。
- Kaggle GPU train v1の15 boosters完走を確認した。
- 380列schema、29列上書き、25列維持、`nsel_*` 0列、15 modelとartifact SHAをlogsで監査した。
- `lgb_mean` 8.101331、同一fold exp238差+0.164641、near / 1000+悪化、改善fold 1/5、worst-well +13.291303によりguard failを確定した。
- replacement-onlyを不採用とし、inference / submitへ進まない判断を記録した。
