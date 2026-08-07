# タスクリスト

## 未着手

- なし。

## 進行中

- なし。

## ブロック中

- なし。

## 完了

- `exp307_finite_only_robust_sigma_gr`として採番し、steeringとexperiment scaffoldを作成した。
- finite std diagnosticとfinite MAD primary、20-pair fallback 30、clip 10--60を固定した。
- exp209 decoder固定、control再実行0、2 variants x 773 = 1,546 HMM runsを固定した。
- performance/long-tail/SHA gateとFAIL後の救済禁止を固定した。
- compact self-contained Jupytext train sourceを実装し、scale freeze、2 HMM variants、late truth join、fixed gateをNotebookセルへ展開した。
- fail-closed inference source/Notebookを実装し、raw-test predictionとsubmissionを無効にした。
- exp209 `_hmm2_fb`をself-contained sourceへ抽出し、AST同一性を確認した。親exp209にはcompact sourceがないため、正規train.pyとの比較でexp209 174行/6章、exp307 1,632行/10章を確認した。
- finite pair、fallback、clip、truth late-join、control/assignment SHA、variant順序、primary gate、disabled inferenceのcontract testを追加した。
- 2 variants、1,546 HMM well-runs、0 booster、control再実行0をconfig/testで再確認した。
- Jupytext test、構文、ruff、対象test、`make validate-exp`、`make validate-template`を通した。
- Kaggle CPU train v1を実行し、1,546 HMM well-runs後のsaved LikPF列契約ミスを記録した。
- `last_known_tvt + likpf_mean_d`復元とHMM前schema preflightを追加し、同一canonical kernel version 2を実行した。
- version 2は3,783,989 rows / 773 wells / 1,546 HMM well-runsを27,402.239秒で完走した。
- finite std / finite MADのdirect、fixed LikPF blend、5 folds、stress scope、by-well、scale audit、SHAを記録した。
- finite MAD primaryのpromotion gate FAILを適用し、救済、inference、submissionなしで閉鎖した。
- exp307 PASSを固定依存にしたexp308--310とexp323--328を未実行のまま閉鎖した。
