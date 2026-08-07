# タスクリスト

## 進行中

- なし

## ブロック中

- なし

## 完了

- [x] exp398を採番し、steeringと実験scaffoldを作成した。
- [x] Route、親、単一変更、保存control、実行量、promotion gateを固定した。
- [x] `docs/06_reproducibility.md`に従うSHA・freeze・CPU設計を記録した。
- [x] compact self-contained train/inference候補と専用testを完成した。
- [x] Jupytext、py_compile、Ruff、pytest 9件、strict validationを通した。

## 実行承認後

- [x] 正規train notebookへcandidateを採用する。
- [x] package metadataとbootstrap内config/sourceの一致を確認する。
- [x] Kaggle private CPU version 1を完了まで監視した。
- [x] logsと必要最小限のoutputからCV、fold、scope、by-well、runtime、SHAを記録した。
- [x] 全gateをAND判定し、`all_well_sigma_x1p3_failed_close_without_rescue`を確定した。
- [x] CSV round-tripのゼロ許容比較による技術監査偽陰性を切り分け、ローカル監査へ
  `atol=1e-12`と回帰testを追加した。科学結果は再実行せずcloseした。
- [x] config / metrics / README / SESSION_NOTES / result /
  experiment_summary / KAGGLE_DIRECTIONを終端状態へ更新した。
