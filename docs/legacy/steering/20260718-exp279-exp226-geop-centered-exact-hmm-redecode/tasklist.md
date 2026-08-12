# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- `docs/06_reproducibility.md` を読み、再現性設計を `design.md` に記入した。
- exp226 OOF `tvt_geop` の exp209固定grid coverageを全3,783,989 rows / 773 wellsで確認した。
- active variant / model config / trained fold / boosterを `1 / 0 / 0 / 0` に固定した。
- parent/control再学習0、GPU/inference/submission無効を固定した。
- self-contained Jupytext train/inference、config、README、SESSION_NOTES、result、metricsを実装した。
- exact forward-backwardのexp270 AST parity、4 unit tests、Ruff、py_compile、Jupytext round-trip、strict experiment validationを通過した。
- template validationとrepository全161 testsを通過し、`experiment_summary.md`へexp279を反映した。
- ユーザー承認scopeを記録し、canonical private CPU kernel version 1をpushした。
- 3,783,989 rows / 773 wellsのKaggle output、全promotion guard、persistent-offset recoveryを監査した。
- input / prediction / decoder / 11 artifact SHAの一致を確認した。
- guard FAILを受け、救済grid・inference・submissionなしでbranchを閉じた。
