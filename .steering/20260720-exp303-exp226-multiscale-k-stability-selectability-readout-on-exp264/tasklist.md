# タスクリスト

## TODO

- [x] exp302 technical PASSとcandidate novelty PASSを確認する。
- [x] corrected parentでのexp276完了とpromotion guard FAILを確認する。
- [x] ユーザーからexp303実装承認を得る。
- [x] Jupytext percent形式の別名compact self-contained readout sourceを作る。
- [x] exp302 predictions、exp264 Stage C v6 candidate score、hidden-like assignmentのSHA preflightを実装する。
- [x] 固定row/H128/boundary featureとprimary H512 scoreを実装する。
- [x] truth-free freezeと別loaderによるlate truth joinを実装する。
- [x] pooled/fold/quintile/1000+/hidden-like/by-well readoutを実装する。
- [x] 専用test、構文、F821、Jupytext変換test、`make validate-exp`を通す。
- [x] Kaggle push前の固定実行量をconfig/sourceでfail-closedにする。
- [x] 正規Notebook placeholderをcompact版へ採用する明示承認を得る。
- [x] Kaggle push直前に`1 readout × 5 evaluation folds`、LightGBM `0`、booster `0`、candidate再生成`0`を再確認する。
- [x] Kaggle package作成・pushの明示承認を得る。
- [x] Kaggle logからinput/feature/schema/score/block/readout SHAを記録する（大きなoutput archiveは不要）。
- [x] PASS/FAIL後に`result.md`、`metrics.json`、`SESSION_NOTES.md`、`experiment_summary.md`、バックログを更新する。

## 完了

- [x] 2026-07-20: `kaggle-strategy`でexp276優先とexp302依存を確定した。
- [x] 2026-07-20: バックログ項目、steering、未実装experiment scaffoldを作成した。
- [x] 2026-07-20: fixed feature schema、primary score、label、PASS/FAIL、freeze順序、禁止事項を確定した。
- [x] 2026-07-20: 1 fixed readout、5 evaluation folds、0 model/boosterを記録した。
- [x] 2026-07-21: exp302 version 2のtechnical PASSとK12/K24 candidate novelty PASSを確認し、prediction SHAを固定した。
- [x] 2026-07-21: exp276 corrected-parent version 3の固定guard FAILを確認し、全dependencyを成立させた。
- [x] 2026-07-21: compact self-contained train/inferenceと専用tests 12件を実装した。
- [x] 2026-07-21: Jupytext round-trip、py_compile、ruff全check、strict experiment validationを通過した。
- [x] 2026-07-21: truth-free input loaderを実ファイルでpreflightし、3,783,989 rows / 773 wells / finite coverage 1.0を確認した。
- [x] 2026-07-21: ユーザーの`実行してください`により正規Notebook採用と1 private CPU readoutのpackage/push/run承認を得た。
- [x] 2026-07-21: canonical private CPU version 1（id_no 128080983）を完了し、technical PASS / scientific FAILを確定した。
- [x] 2026-07-21: pooled AUC 0.488805、AUC>0.5は1/5 folds、全stress scope逆方向のため救済gridなしでbranchを閉じた。
