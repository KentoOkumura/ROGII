# タスクリスト

## TODO

- なし。temperature-only direct PF update は train-side 不採用として終了した。

## 進行中

- なし。

## ブロック中

- なし。

## 完了

- temperature と outlier mixture を別実験として扱う方針を確定した。
- temperature-only の変数、guard、再現性設計を固定した。
- v1 の exp072 cache control 欠落を記録し、ユーザー指定の exp209 復元経路を採用した。
- v3 の timeout / `CANCEL_ACKNOWLEDGED` を記録し、ユーザー承認により variant 別 split run を採用した。
- `temp_t2` v1 と `temp_t4` v2 を別 CPU kernel で完走した（各 773 wells、3,783,989 rows、0 boosters）。
- `temp_t2` は RMSE 13.529887（control比 +1.934989）、`temp_t4` は 13.532730（+1.937833）。
  1000_plus と worst-well も大きく悪化したため、両方を不採用とし、inference / submit を行わない。
- gate 後の long-lived path regression を先に監査する containment audit を、将来の robust-likelihood 再検討時の前提にした。
