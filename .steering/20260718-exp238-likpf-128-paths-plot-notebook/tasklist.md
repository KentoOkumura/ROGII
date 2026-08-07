# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- なし。

## 完了

- steeringを作成した。
- 再現性設計を`design.md`に記入した。
- stable per-well seed、thread並列、軌跡一時保持、saved mean parity方針を固定した。
- Jupytext percent sourceと別名`.ipynb`を実装した。
- PF 128本をalpha 0.06で描画し、true TVTとexp238 LightGBM OOFを不透明な太線で前面表示するcontractを実装した。
- CPU/internet-off、別canonical slugのKaggle packageを作成した。
- `py_compile`、Ruff、Jupytext `--test`、JSON/cell/metadata監査、strict experiment validationを通した。
- `SESSION_NOTES.md`へ実装内容と未実行状態を記録した。
- ユーザーがKaggle実行を明示依頼した。
- 実行variant 1、model config 0、fold学習0、booster 0、parent/control再学習なしを確認した。
- canonical kernel version 1が`COMPLETE`になったことを確認した。
- 3,783,989行 / 773 wells、773 plots、各well 128 seeds × 500 particlesをsummaryとmanifestで監査した。
- 128-seed平均と保存済みPF平均が773 / 773 wellsでexact parity、最大差0.0であることを確認した。
- 代表図でPF 128本、truth、exp238 LGB OOF、alpha 0.06の描画を目視確認した。
- summary、manifest、代表PNG、kernel logをローカル履歴へ取得し、`SESSION_NOTES.md`、`result.md`、`metrics.json`を更新した。
