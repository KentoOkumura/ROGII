# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- なし。

## 完了

- steering 作成。
- exp217 実験ディレクトリ作成。
- config / helper / train notebook / inference notebook / README / result / metrics placeholder を exp217 用に更新。
- 再現性設計を `design.md` に記入。
- Kaggle train v2 GPU rerun 前 guard として 15 boosters / control retraining なしを `SESSION_NOTES.md` に追記。
- Kaggle train v2 を T4 GPU metadata と `--accelerator NvidiaTeslaT4` で push し、`KernelWorkerStatus.RUNNING` を確認。
- v2 が `KernelWorkerStatus.CANCEL_ACKNOWLEDGED` で停止し、CV が未生成であることを確認。
- `pubraw_` generation を `pfbeam_features` notebook に分離し、通常 train は cache kernel source として読む構成に変更。
- cache notebook / train notebook の Kaggle package 生成と metadata 確認。
- 初回 cache push は `Maximum batch CPU session count of 5 reached` で拒否されたため、同内容を `kentookumura/exp217-pubraw-cache-v1` として push し、`KernelWorkerStatus.RUNNING` を確認。
- `kentookumura/exp217-pubraw-cache-v1` が `KernelWorkerStatus.COMPLETE`。3,783,989 rows / 773 wells / 25 pubraw features を生成。
- `kentookumura/exp217-grcal-pubraw-pf-conf-exp158-train` v3 を CPU / pubraw cache source 付きで push し、`KernelWorkerStatus.RUNNING` を確認。
- train v3 が `KernelWorkerStatus.COMPLETE`。best RMSE `10.669620824` で exp158 からは改善したが exp184/exp191 に届かないため、inference / submit へ進めない判断を記録。
- 2026-07-14 に closeout を記録し、`KAGGLE_DIRECTION.md` の状態を `完了・不採用` に更新。
