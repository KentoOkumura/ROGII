# タスクリスト

## TODO

- fle3n / SP45 / Koolbox 系の exact source slug を固定して追加監査する。
- `pilkwang_branch_decomposition` を別実験または後続タスクとして切り出す。

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering requirements / design / tasklist を作成した。
- `config.yaml` に Pilkwang、ridge-sp、fle3n/SP45/Koolbox placeholder、anchor submission path を定義した。
- `public_artifact_integrity_audit.py` を追加した。
- train / inference notebook を no-submit integrity audit entrypoint に置き換えた。
- 再現性設計を `design.md` と `config.yaml` に記入した。
- Kaggle train notebook を `kentookumura/exp079-public-artifact-audit-train` v4 として実行し、実際の `/kaggle/input` source と branch output を監査した。
- output 取得後に `SESSION_NOTES.md`、`result.md`、`metrics.json` を更新した。
- 監査結果に基づき、`KAGGLE_DIRECTION.md` の公開 notebook route backlog を更新した。
