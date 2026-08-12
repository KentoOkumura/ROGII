# タスクリスト

## TODO

- なし。Stage 0科学guard不通過によりbranchを閉じた。

## 進行中

- なし。

## ブロック中

- なし。

## 完了

- 物理モデル単独LB 6.5目標とoracle禁止要件を固定した。
- `F = Z + TVT + c_w`に基づくanchor-relative prediction式を固定した。
- Stage 0 / Stage 1 / 条件付きStage 2の順序、成功条件、停止条件を固定した。
- outer-valid formation/true suffix除外とfold-batch transductive CV契約を固定した。
- fault-aware truncated-quadratic graph、deterministic IRLS、単一MAP出力のmodel contractを固定した。
- 再現性、SHA、runtime、leakage、CV/LB、oracle禁止guardを設計へ記録した。
- `exp289`のdesign-only experiment scaffold、config、README、SESSION_NOTES、result、metricsを作成した。
- `KAGGLE_DIRECTION.md`の未着手バックログ最上位と`experiment_summary.md`へ記録した。
- `make validate-exp EXP=exp289_fault_aware_transductive_geological_potential`をPASSした。
- 2026-07-19の追加実装承認を記録した。
- Stage 0だけを実装するJupytext percent形式のcompact self-contained train sourceを作成した。
- disabled inference sourceを作成し、Stage 1別承認前はfail-closedにした。
- outer-valid formation/true suffix除外、truth-after-freeze、graph identity、stable edge order、formation identity、欠損source処理、target concatの専用tests 9件を作成した。
- Stage 0の1 audit variant、ML config 0、trained fold 0、booster 0、control再生成0を再確認した。
- compact sourceを正規train/inference notebookへ反映し、Jupytext `--test`、`py_compile`、ruff、専用pytest、strict experiment validationを通した。
- Kaggle packageのmetadataとbootstrap parityを確認し、Stage 0 CPU v1をcanonical kernelへpushした。
- v1が全行欠損source ANCCで停止したことを確認し、solver/booster未実行を記録した。
- 全行欠損source ANCCをfold-safeに除外する技術修正を検証し、同じcanonical kernelへCPU v2をpushした。
- v2がsource node構築を通過後、target-node concat attrs errorで停止したことを確認し、solver/risk未実行を記録した。
- target-node concat attrsの技術修正を回帰検証し、同じcanonical kernelへCPU v3をpushした。
- Stage 0 CPU v3を241.548秒で完了し、technical guard全PASS、AUC 0.570652、Spearman 0.127885、正方向5/5 foldsを確認した。
- 事前AUC/Spearman guard不通過により`close_branch_without_rescue_grid`を適用した。
- 必要なv3 outputだけを取得し、graph/node/well frozen SHA、7生成物manifest、773 well coverage、pushed config/source SHAを照合した。
- Stage 1/2 solver、inference、submissionを未実装・未実施のまま維持した。
