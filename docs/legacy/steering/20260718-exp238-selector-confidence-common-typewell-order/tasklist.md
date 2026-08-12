# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- なし

## 完了

- steeringを作成した。
- 再現性設計を `design.md` に記入した。
- stochastic処理、model fit、candidate再生成、submission生成を追加しない方針を固定した。
- exp065 common typewell対応表の解決・filter・coverage guardを実装した。
- PNG名、manifest、zipを共通typewell順へ揃えた。
- summaryにordering contractと対応表SHAを追加した。
- Jupytextから正規notebookを再生成し、Kaggle packageを同期した。
- metadata/configへexp065 kernel sourceを追加した。
- `py_compile`、`ruff --select F821`、Jupytext `--test`、strict validationを通した。
- Kaggle push前にmetadataとconfigの7 kernel sourcesが一致することを確認した。
- 同じcanonical kernelへversion 3としてpushし、post-push source/metadataを確認した。
- ユーザー指示に従い、`RUNNING`確認後の監視を停止した。
- ユーザーの完了連絡後、version 3が`COMPLETE`であることを確認した。
- Kaggle files APIの773 PNGが共通typewell順であることを確認した。
- manifestをexp065対応表と全件照合し、773 wells / 54 groupsの順序とcoverageを確認した。
- plots zipのcentral directoryを監査し、773 membersがmanifest順と一致することを確認した。
- 先頭PNGを目視し、plot内容とv2配色が維持されていることを確認した。
- global metricsとselector distributionがv2から不変であることを確認した。
- `SESSION_NOTES.md`、`result.md`、`metrics.json`へ確定結果を記録した。
