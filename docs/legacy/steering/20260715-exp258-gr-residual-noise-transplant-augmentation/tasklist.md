# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- なし。

## 完了

- 要件、stage、fold-safe donor、再現性、停止guardの設計を記録した。
- exp238 fold/candidate/rank-slot/final model contractを必要部分だけnotebookへ移植した。
- residual affine fit、block inventory、stable donor selection、missing mask、white/shuffled controlを実装した。
- synthetic contract testでreconstruction、donor isolation、stable SHA、donor順序非依存を確認した。
- Stage 0 residual auditと生成物保存を実装した。
- Stage 1 nested ranker trainingとclean validation readoutを実装した。
- Stage 2 hard guardとfinal TVT LightGBM 15-booster trainingを実装した。
- inference停止guardと将来のclean raw-test saved-model契約を実装した。
- selector train / final train / inferenceをJupytext percent形式からipynbへ変換した。
- py_compile、ruff F821/F401、Jupytext `--test`、strict `validate-exp`、pytestを通した。
- `SESSION_NOTES.md`へstage別variant/config/fold/booster数とcontrol再学習なしを記録した。
- `experiment_summary.md`と`KAGGLE_DIRECTION.md`を実装待ち状態へ更新した。
- selector / final停止状態packageのmetadata、artifact source、bootstrap dependency、config SHA、
  CPU/GPU・internet・run-on-push設定を確認した。
- Kaggle selector v1でStage 0 auditとprimary ranker 20 boostersを完走した。
- residual audit、20 model manifest、clean validation、historical exp238比較とSHAを監査した。
- selector guard 5/6 failを記録し、conditional final 15 GPU boostersを実行しない判断を確定した。
- final/inference/submission未実行のためprediction/submission SHAが存在しないことを記録した。
