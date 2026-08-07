# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- なし

## 完了

- `docs/06_reproducibility.md`を確認し再現性設計を`design.md`へ記入した。
- PF dynamics 2 variants、LightGBM 0 config、fold 0、booster 0、親/control再学習なし、GPUなしの実行契約をユーザー承認済み。
- exp266実験ディレクトリ、config、self-contained Jupytext train、disabled inferenceを作成した。
- exp072 PF ANCC / PF-Z exact kernelとstable seed契約を抽出した。
- 全well seed別指標、nested aggregate、path quantile、parity、発生条件readoutを実装した。
- static validation、初版3 contract tests、Jupytext round-trip、strict experiment validationを通した。
- Kaggle v1の参照coverage fail-closedを診断した。3,783,989 rowsは一致したがmixed dtypeにより776 wellsとなった。
- exp072 `id` / `well`、exp209 `id`、exp226 `well_id`をCSV parse時からstring固定した。
- `id` prefixと`well`の全行一致guard、数字型well IDの回帰testを追加し、全4 testを通した。
- canonical / package / bootstrap SHA一致を確認し、同一Kaggle kernelへCPU notebook v2をpushした。
- v2 pullでid_no `127531300`、private CPU、3 kernel sources、1 competition source、修正guard反映を確認した。
- Kaggle v2は参照coverage 3,783,989 rows / 773 wellsを通過したが、最大0.000484 ftのfloat32保存差でparity停止した。
- 親exp072とexp106のdtype経路を確認し、参照`pf_ancc` / `pf_z`をfloat32へ復元する修正とtestを追加した。
- canonical / package / bootstrap SHA一致を確認し、同一Kaggle kernelへCPU notebook v3をpushした。
- v3 pullでid_no `127531300`、private CPU設定、float32 parse/dtype guard、well identity guard反映を確認した。
- Kaggle v3を3時間28分02秒で完走し、両PFのseed 0全行exact parityを確認した。
- 98,944 seed rows、27,828 aggregate rows、1,546 stability rows、494,204 strong path rowsを取得した。
- 必須12 artifactsのbytes、raw SHA、decompressed SHAを全件照合した。
- `11d0f5ac`のrobust判定、strong 53 wellsの再現率、発生条件、seed集約収束を解析した。
- `result.md`、`metrics.json`、`SESSION_NOTES.md`、`README.md`、全体backlogを完了状態へ同期した。
