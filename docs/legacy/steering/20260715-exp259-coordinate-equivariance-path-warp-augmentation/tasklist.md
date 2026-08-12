# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- `train_exact_datum_after_transform_audit`を別名Jupytext notebookとして実装し、既存audit notebookを履歴として保持した。
- exp259ではcontrol/parentを再学習せず、exp251 control未完了時は比較だけpendingにしてOOFを保存する契約を固定した。

- `docs/06_reproducibility.md`を確認し、stable seed / SHA / Kaggle bootstrap方針を設計へ記録した。
- exp251を参照親としてexp259 steering/scaffoldを作成した。
- reusable coordinate/path augmentation engineを`src/`へ実装した。
- strict inverse consistency、approximate anchor continuity、GR resample、candidate equivariance、distribution rejectのunit test 13件を追加した。
- compact Jupytext train/inference notebookと正規`.ipynb`を作成した。
- config、README、SESSION_NOTES、result、metrics、experiment summary、KAGGLE_DIRECTIONを更新した。
- Jupytext convert/test、py_compile、Ruff、exp259 unit test、strict validate-exp、validate-templateをpassした。
- full pytestはexp259を含む45件PASS、既存exp251 configのstageがtest期待`feature_audit_only`ではなく`train_after_feature_audit`のため既存1件のみFAILした。exp259起因ではないためexp251は変更していない。
- canonical Kaggle CPU train packageを生成し、GPU/internet false、competition source、run-on-push、root/package config一致、embedded `planned_boosters: 0` / `transform_audit_only` / `src/coordinate_path_augmentation.py`を確認した。
- Kaggle support bundleへ`src/__pycache__`/`.pyc`が混入しないよう共通prepare scriptのfilterを修正し、生成packageにbytecodeが含まれないことを確認した。
- Kaggle CPU version 1で773 wells、9 transforms、6,957 viewsの0-booster auditを完了した。
- strict 4変換は全件通過し、inverse最大誤差`9.313225746154785e-10`、local metric相対差最大`0.0`を確認した。
- 近似変換4種は94.8%以上を採択し、`md_stretch`はreal-train geometry envelope違反で全件rejectされたことを確認した。
- summary、envelope、manifest、preview、real-well summaryのraw/decompressed SHAをローカル再計算し、Kaggle summaryと一致した。
- model / prediction / submissionは生成していないため各SHAは対象外であり、deterministic submission anchorとして扱わない。
- exp259 exact datum version 1とexp251 corrected 295列controlの`COMPLETE`を確認した。
- 両runのsaved metrics/by-wellを比較し、overall / candidate logloss / 1000+はPASS、
  hidden-like 2面 / 最大well回帰はFAILと確定した。
- exp259をtrain-side rejectedとし、inference / submissionへ進めない判定を記録した。
- 必要なsmall artifactsだけを取得し、summary/schema/manifest/OOF content SHAを記録した。
- 追加監査でOOF、10 models、5 imputersを`/tmp`へ選択取得し、診断15/15・model 10/10・imputer 5/5・OOF decompressed SHAの独立一致を確認した。
- exp251と同じ5 foldsを再構成し、exp259は4/5 folds改善、fold 0のみ+0.066196悪化と確認した。
