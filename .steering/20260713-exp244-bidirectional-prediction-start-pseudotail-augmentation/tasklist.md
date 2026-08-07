# タスクリスト

## TODO

- なし。v4 guard失敗によりbranchを停止する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering docs作成。
- 再現性設計を `design.md` に記入。
- exp244実験ディレクトリ作成。
- early/original/late view manifestとguard実装。
- startごとの`TVT_input`再構成とanchor/prefix feature materialization実装。
- replay contract、distribution readout、schema/content SHA保存実装。
- current-test known-prefix calibration request audit実装。
- self-contained train/inference Jupytext notebook作成とipynb変換。
- static check、F821、strict experiment validation pass。
- Kaggle train/inference package prepareとmetadata/bootstrap config確認。
- template validationとpytest 13件pass。
- `SESSION_NOTES.md`、`result.md`、`metrics.json`、`KAGGLE_DIRECTION.md`更新。
- Kaggle CPU train audit v1完了。773 wells / 3,854 views / 3,850,880 rows、全guard pass。
- Kaggle CPU inference calibration audit v1完了。3 wells / 6 requests / 3,750 rows、全guard pass。
- train/inference content SHA、kernel ID/version、slug復旧履歴を記録。
- exp218 train v1 outputを取得し、OOF 3,783,989 rows / 773 wells / RMSE 8.475793978、decompressed SHA `5f3fc951...2976`をローカル事前監査。
- ローカル規則再構成では155 wells一致 / 618 wells不一致だったが、tie順序の環境差があるため参考値へ降格。
- Kaggle guard v2でexp244 v1実fold manifestを直接比較し、174 wells一致 / 599 wells不一致を確定。
- frozen-anchor parity guard v2完了。OOF/raw surface/model manifest/v1 fold SHAの全guard pass、0 booster。
- v3 dual-start confidence-shrinkをsingle pre-registered variant / CPU / 0 boosterで実行。
- v3はoverall +0.001449、1000+ +0.001200、hidden-like 2面 +0.000602程度、改善fold 1/5でguard failed。不採用。
- v3 current-test port、threshold/max-shrink grid、submissionを停止。
- v4 direct integrated trainingの要件・行数・cache分割・GPUコストを確定。
- offset別4本のfull 380-feature cache Jupytext notebookを実装。
- exp239 official全行と4 pseudo cacheをmemmapで統合するGPU train notebookを実装。
- outer-valid source well除外、frozen exp218 OOF比較、overall/1000+/hidden-like/fold/by-well guardを実装。
- raw 773 wellsで4 offsetの3,081 views / 770,157 rows契約を再計算し、設定値との完全一致を確認。
- v4 5 notebooksのJupytext test、ruff、py_compile、strict experiment validationをpass。
- CPU cache 4 packageとGPU train 1 packageをstrict prepareし、metadata/bootstrap入力を確認。
- repository template validationとpytest 15件をpass。
- v4実装内容と未実行状態をREADME、SESSION_NOTES、result、metricsへ記録。
- offset別4 CPU cache v1を実行し、合計3,081 views / 770,157 rows / 124 shardsと共通380-feature schemaを確認。
- 4 cacheのmanifest/request/schema/feature SHAを記録し、configへv1 SHAをpin。
- 1 variant / 3 configs / 5 folds / 15 boosters、control再学習なしの明示承認を記録。
- `kentookumura/exp244-bidirectional-multiview-train` v1をGPUでpushし実行開始。
- GPU train v1完了。OOF 8.472379731、raw exp218比-0.003414021。
- overall、1000+、hidden-like 2面、3/5 folds改善guardはpass。
- 14 wellsが+2 ft超悪化し、worst `059c8f24` +16.650567でworst-well guard fail。
- `adoption_supported=false`を記録し、inference / submission / mixed-weight gridを停止。
- metrics / by-well / summaryだけをselective downloadし、artifact SHAを記録。
- `README.md`、`SESSION_NOTES.md`、`result.md`、`metrics.json`、`KAGGLE_DIRECTION.md`を完了状態へ更新。
