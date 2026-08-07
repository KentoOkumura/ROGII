# 設計

## アプローチ

`scripts/public_notebook_catchup.py` を追加し、公開 notebook archive を横断して inventory を生成する。

入力:

- `docs/notebooks/rogii-wellbore-geology-prediction/{score_ascending_latest,vote_top,date_run_recent}/kernel_listing.csv`
- 各 notebook directory の `kernel-metadata.json`
- metadata に紐づく `.ipynb` / `.py`

出力:

- `public_notebook_catchup_after_self_improvements_<date>.md`
- `public_notebook_catchup_inventory_<date>.csv`

処理:

- listing rank、vote、lastRunTime、title score を統合する。
- metadata から GPU/internet/dataset/kernel/model sources を取得する。
- code text から PF、beam、DWT、TabICL、AeroRidge、SG smoothing、formation/geology、visible branch、static submission などの signal を検出する。
- target ref と score/vote rank から replay priority を付ける。
- first replay candidate と artifact-stack audit candidate を分ける。

## 実験範囲

- 対象実験: なし。replay 実験を切る前の catch-up 実装。
- 親実験: `exp026_pseudo_tail_bucket_shrink_inference_submit` を self anchor とする。
- 変更する変数: public notebook inventory の生成ロジック、catch-up report。
- 固定する変数: 自前 CV/LB anchor、既存 experiment code、既存 notebook archive。

## リスク

- リークリスク: 公開 notebook が formation/geology、public visible branch、static CSV blend を含む可能性がある。risk flag として棚卸しし、replay 後に採用可否を判断する。
- CV/LB 不一致リスク: 公開 LB 8.86 系は hidden rerun / artifact / branch 条件が不明なため、自前 CV と直接比較しない。
- ランタイム/メモリリスク: PF seed ensemble や TabICL artifact stack は実行時間と GPU/input 固定が問題になるため、metadata で先に切り分ける。
