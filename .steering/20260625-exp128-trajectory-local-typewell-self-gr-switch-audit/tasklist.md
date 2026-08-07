# タスクリスト

## TODO

- `task validate-exp EXP=exp128_trajectory_local_typewell_self_gr_switch_audit` を通す。
- Kaggle train notebook を prepare / push する。
- output 取得後に feature content SHA、OOF prediction SHA、kernel version を記録する。
- candidate metrics、window diagnostics、worst-well regression を読んで raw-test parity audit へ進めるか判断する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering docs を作成した。
- exp128 実験フォルダを作成した。
- 再現性設計を `design.md` に記入した。
- `trajectory_local_typewell_self_gr_switch_audit.py` を実装した。
- train / inference notebook を診断用のセル構成に更新した。
- `config.yaml`、README、SESSION_NOTES、result、metrics を `implemented_not_run` として更新した。
- 新規処理は RNG 不使用、single-process とし、global RNG / thread scheduling 依存を入れていない。
