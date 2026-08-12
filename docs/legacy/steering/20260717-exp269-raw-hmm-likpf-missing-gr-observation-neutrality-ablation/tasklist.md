# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- なし

## 完了

- scientific parentをexp209 raw exact HMM、artifact ancestorをexp205、implementation referenceをexp247と確認した。
- `docs/06_reproducibility.md`を確認し、Stage 1の再現性設計を記入した。
- exp269 experiment scaffoldとsteeringを作成した。
- exp209/exp205 fixed-control resolverとdecompressed SHA guardを実装した。
- raw exact HMMのGR missing-row observation neutrality 1変更を実装した。
- raw train/test missing inventoryとpaired group/by-well/posterior/divergence診断を実装した。
- synthetic emission contractとfail-close Stage 1 guard判定を実装した。
- readableなJupytext train/no-inference sourceと通常`.ipynb`を作成した。
- README、SESSION_NOTES、result、metrics、experiment_summaryをimplemented状態へ更新した。
- Jupytext、py_compile、Ruff F821/F401/E9、targeted test 6件、strict experiment validationを通した。
- Kaggle CPU train packageをstrict生成した。
- package metadataとbootstrap configでCPU/GPU/internet、kernel source、active variant、学習コスト、PF/inference disabledを照合した。
- ユーザー承認後、canonical kernel version 1をKaggleへpushし、id_no `127592556`とCPU metadataを確認した。
- Kaggle CPU Stage 1 version 1の`COMPLETE`を確認し、logsと必要な小規模成果物からscore、runtime、input/output SHAを記録した。
- overall RMSE 11.938287 -> 13.348499（`+1.410212 ft`）、missing rows `+2.548257 ft`、observed rows `+0.846115 ft`を確認した。
- hidden-like 2面、1000+、worst-wellを含む事前固定guardが不通過、finite coverageとID整合は通過した。
- `pf_stage_eligible=false`としてlikelihood-PF Stage 2をfail-closeし、inference / submissionを実行せずbranchを終了した。
- `metrics.json`、`result.md`、`README.md`、`SESSION_NOTES.md`、`experiment_summary.md`、`KAGGLE_DIRECTION.md`へ完了判断を反映した。
