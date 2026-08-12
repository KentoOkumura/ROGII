# タスクリスト

## TODO

- なし。peer atlas emissionの直接加算はtrain-side不採用としてclosed。

## 進行中

- なし

## ブロック中

- なし

## 完了

- exp231 steering と実験フォルダを作成した。
- fold-safe typewell peer atlas、state-centered score、target-free confidence gate、3 alpha HMM variantsを実装した。
- cluster assignment / atlas fold summary / feature content SHA の記録を実装した。
- direct comparison に true-state rank と persistent-offset onset AUC/q90 lift を追加した。
- Python構文、F821 lint、Jupytext conversion/test、strict validate-exp、template validationを通した。
- v1 timeoutを分析し、unique-bin marginとrelative-fit gateへ修正した。full atlasを維持する12 target-well preflight modeを追加した。
- V2 preflightのKaggle packageをstrict生成し、bootstrap内config と kernel metadata のunique-bin gate、12 target wells、1 variant、full atlas設定を確認した。
- Kaggle CPU v2 preflightを完了した。12/12 target wellsでgateは非zero（平均0.077299）、HMM平均48.293秒/well、full run見積もり約5.3時間を確認した。direct comparison/CV判定はまだ行っていない。
- Kaggle CPU v3正式full runを完了した。全773 wells / 3,783,989 rowsでgateは有効、global RMSEはexp072 likPF比 `-0.024947` だったが、`1000_plus`は `+0.016570`、316 wells悪化、最大well悪化 `+48.316178`、offset-onset AUC `0.507654`、hidden-like未評価のため不採用。raw-test / inference / submitは行わない。
