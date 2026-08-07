# タスクリスト

## TODO

- Kaggle train を push する場合は、metadata と bootstrap 内 config の整合を確認する。
- Kaggle train 完了後、OOF metrics、path switch、worst-well、distance bucket、exp115 subgroup、feature importance を `SESSION_NOTES.md` / `result.md` / `metrics.json` / `experiment_summary.md` に記録する。
- 結果が positive でも、raw-test parity / hidden-like stress / worst-well guard を見ずに inference port / submit しない。

## 進行中

- なし

## ブロック中

- なし

## 完了

- 再現性設計を `design.md` に記入した。
- exp183 の config / module / Jupytext train・inference source を作成した。
- train 予定コストを active variant 1、LightGBM config 3、fold 5、booster 15、control 再学習なしとして設計した。
