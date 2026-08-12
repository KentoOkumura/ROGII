# タスクリスト

## 目的

exp287のformation global gainとwell-level tail regressionを分けるtarget-free risk familyがあるかを、0-booster readoutとして設計固定する。

## 変更点

- 既存の未採番候補をexp336へ切り出し、モデル変更なしのStage A/B attribution契約として固定する。

## 未着手（別途承認が必要）

- なし。

## 進行中

- なし。

## ブロック中

- なし。inferenceとsubmissionは未承認ではなく実験契約上の禁止を維持する。

## 完了

- [x] exp334 tail FAILにより候補の再開条件が成立したことを確認した。
- [x] 未使用番号`exp336`を採番した。
- [x] steeringをexperiment scaffoldより先に作成した。
- [x] `exp336_exp287_formation_tail_attribution_readout` scaffoldを作成した。
- [x] Routeを`ml_model`診断、親をexp287、比較をcorrected exp264、triggerをexp334に固定した。
- [x] 6 primary risk familyとtarget-free aggregation/risk方向を固定した。
- [x] Stage A freeze SHA後だけStage B OOF error joinを許可するleakage boundaryを固定した。
- [x] global effect、median、4/5 folds、hidden-like 2面、coverageのAND gateを固定した。
- [x] model/config/fold/booster/control再学習をすべて0に固定した。
- [x] input artifact SHA、canonical order、RNGなし、Kaggle bootstrap方針を記録した。
- [x] 同一OOF救済、formation列削除、corrected prediction、inference、submissionを禁止した。
- [x] config、README、SESSION_NOTES、result、metricsをdesign-onlyへ更新した。
- [x] `KAGGLE_DIRECTION.md`の候補をexp336 design-frozenへ更新した。
- [x] `experiment_summary.md`へexp336を追加した。
- [x] ユーザーの明示依頼によりimplementationとcompact self-contained Notebook候補を承認済みに更新した。
- [x] Stage Aへvalid 5 partitionのSHA/schema/row/well/finite/identity監査と6 familyの固定well集約を実装した。
- [x] raw contextを`MD/X/Y/Z/TVT_input`だけで構成し、target-free属性、四分位、canonical CSV SHA、freeze manifestを実装した。
- [x] Stage Bをfreeze SHA検証後だけOOFを開くAPIに分離し、ID/well/fold/actual照合とwell等重みRMSE deltaを実装した。
- [x] global/fold/hidden-like/coverageの固定AND gate、report-only指標、全11生成物、再現性manifestを実装した。
- [x] inference候補をprediction/submission禁止のfail-closed entrypointとして実装した。
- [x] 禁止列、family scalar、known-prefix定数、raw context、freeze改ざん、quartile eligibility、固定gate、OOF整合のsynthetic tests 10件を追加した。
- [x] compact train/inferenceのJupytext sourceと`.ipynb`候補を生成した。既存canonical Notebookは上書きしていない。
- [x] ユーザーの`実行してください`をcanonical採用と1回のKaggle CPU Stage A/B run承認として記録した。
- [x] 6 families / model 0 / LightGBM config 0 / trained fold 0 / booster 0 / control再学習0をpreflightし、canonical Notebookとpackageを生成した。
- [x] Kaggle kernel slug長による作成前400を、意味を維持した50文字未満slugへ修正した。
- [x] version 1のSHA同一assignment複製によるtechnical ERRORを、科学契約不変のdeterministic equivalent-copy resolverで修正した。
- [x] resolver test追加後、専用11 tests、共通込み15 tests、Jupytext、py_compile、ruff F821、strict validationをPASSした。
- [x] Kaggle private CPU version 2（id_no `128221753`）を完了した。runtimeはreadout本体`92.458 sec`。
- [x] 11成果物を取得し、記録されたartifact SHAを実ファイルと照合して全一致した。
- [x] 6 familyを固定AND gateで評価し、passed family `0/6`、`NO_STABLE_FORMATION_ATTRIBUTION_CLOSE`を確定した。
- [x] prediction、inference、submissionを生成せず、formation attribution枝を閉じた。
- [x] config、README、SESSION_NOTES、result、metrics、experiment summary、strategy backlogを完了状態へ更新した。

## 次のアクション

- なし。同じOOFでのfamily/threshold救済を行わずcloseを維持する。
