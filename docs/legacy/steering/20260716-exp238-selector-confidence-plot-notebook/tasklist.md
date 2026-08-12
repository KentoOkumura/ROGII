# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- なし。

## 完了

- steeringを作成した。
- 再現性設計を `design.md` に記入した。
- stochastic処理・PF/Beam再生成・model fitが0である方針を固定した。
- Jupytext percent sourceと正規 `.ipynb` を実装した。
- selector top-1 category strip、top2−top1 margin、exp238 OOF比較、manifest/summary出力を実装した。
- `py_compile`、`ruff --select F821`、Jupytext `--test`、JSON parse、strict experiment validationを通した。
- CPU/internet-off、6 kernel sourcesのcustom Kaggle packageを作成した。
- canonical/package notebookの16 cell source一致、output 0、execution count 0を確認した。
- `SESSION_NOTES.md`へ実装内容と未実行状態を記録した。
- ユーザーの明示依頼後、`run_on_push=true`でcanonical kernel v1をpushした。
- Kaggle v1の`COMPLETE`、3,783,989行のstrict outer-valid coverage、773 / 773 well plots生成を確認した。
- summary、manifest、top-1 distribution、artifact SHAを検証し、代表図を目視確認した。
- `result.md`、`metrics.json`、`SESSION_NOTES.md`へ実行結果を記録した。
- exp083 v12と共通するML OOF、PF ANCC、Beam、LikPF、exp226、HMMの色を参照元と一致させた。
- selector top-1 pathを灰色破線にし、共通candidateのtop-1色帯もexp083色へ揃えた。
- 修正後notebook/packageのJupytext、構文、F821、strict validation、配色contractを検証した。
- 同じcanonical kernelのv2をpushし、pull後sourceのセルと色コード一致を確認した。
- Kaggle v2の`COMPLETE`、773 / 773 plots、v1との数値一致を確認した。
- v2 summaryの`plot_colors`、manifest、distribution、SHAを検証した。
- v2代表PNGを取得し、exp083配色とselector色帯を目視確認した。
- v2を正として`result.md`、`metrics.json`、`SESSION_NOTES.md`を更新した。
