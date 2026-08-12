# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- 公開 notebook とパラメータを `cdeotte/xgb-starter-cv-15` version 3 に固定。
- 再現性設計を `design.md` に記入。
- 実行プランを1 variant / 1 config / 5 folds / 5 boosters / parent-control再学習0に固定。
- compact self-contained train notebookとdisabled inference notebookを実装し、canonical `.ipynb`へ変換。
- `config.yaml`、`SESSION_NOTES.md`、`result.md`、`metrics.json`、`README.md`を更新。
- `KAGGLE_DIRECTION.md`の実装済みbacklogをtrain待ち表へ移動。
- `experiment_summary.md`を`implemented_not_run`へ更新。
- Jupytext round-trip、構文、Ruff全体 / F821、固有5 tests、repository 124 tests、strict experiment validationを完了。
- canonical T4 packageを`run_on_push=false`で作成し、metadata、bootstrap 30 entries、公開source SHA、parameter parityを確認。
- ユーザー承認後に`run_approved=true`へ変更し、T4 / run-on-push package、config byte parity、公開source SHA / parameter parityを再確認。
- `kentookumura/exp275-xgb-final-regressor-exp238-train` version 1をpushし、server側T4 / internet off / RUNNINGを確認。
- version 1のapproval文字列不一致をconfig contract mismatchとして特定。データ読込前、booster 0本を確認。
- approval値をguard要求値へ統一し、configとnotebook guardの一致を検証する回帰テストを追加。
- version 2 packageの固有5 tests、strict validation、bootstrap parityを確認し、同じkernel IDへpush。version 1停止点を越えてRUNNINGを確認。
- Kaggle T4 train version 2を5/5 boosters、2,250 trees、elapsed 2,984.807秒で完走。
- logs / cell outputからoverall、fold、1000+、hidden-like、by-well、固定blend、raw guardを確認。
- 一時取得したoutputで3,783,989行OOF、5 model SHA、fold matrix SHA、主要artifact SHAを検証。
- raw XGBoost 8.302528478、parent比+0.365838447、改善0/5 folds、全raw guard FAILを記録。
- inference / submit / parameter rescueを不採用とし、`result.md`、`metrics.json`、`experiment_summary.md`、`KAGGLE_DIRECTION.md`を完了状態へ更新。
- reference inference notebookを917行 / 8章で実装し、Jupytext / syntax / Ruff / contract tests / strict validationをPASS。
- repository全128 testsをPASS。
- canonical T4 inference packageのmetadata、bootstrap 30 entries、config parity、10 kernel sources、0 training契約を監査。
- inference version 1の`position` / `feature_index` schema差を、モデルload・current-test再生成前のcontract ERRORとして特定。
- exp275固有`feature_index`連番guardへ修正し、静的検証を再PASS。
- Kaggle T4 reference inference version 2を415.815秒、14,151行、fallback 0で完了。
- 5 XGBoost / 15 parent / 20 selector model、415特徴、fold matrix SHA、submission SHAを監査。
- raw `submission.csv`をsample互換、行数、ID順、重複、finite、SHAでsubmit-checkし、FAIL/WARN 0を確認。
- raw XGBoostだけを1件submit。submission ref `54798185`として提出記録を固定。
- submission ref `54798185`の`COMPLETE` / Public LB `7.760`をKaggle APIで確認し、全実験記録へ反映。
- monitorが追従した追加ref `54798337`も340分で`COMPLETE` / 7.760。正規記録はSHA追跡済みref `54798185`とした。
