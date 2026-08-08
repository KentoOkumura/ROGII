# タスクリスト

## TODO

- v2 compact self-contained train / inference候補を実装する。
- 5 PF feature schema、PF runtime override、tabular model count、model manifestをcontract testで固定する。
- local static / Jupytext pairing / small-well smokeを通す。
- Kaggle trainで5 PF feature生成と25 base models + 5 Ridge foldsを実行する。
- CV `7.50`再現gateを判定し、PASSした場合だけinference / late submitへ進む。

## 進行中

- v1の契約不一致を保持した同一exp内v2修正。

## ブロック中

- なし。

## 完了

- ユーザー指摘により、名称変更ではなく同一exp内でstage 2-2 systemへ修正する方針を確定した。
- 公開discussion、公開v96 config、公開Ravaghi tabular notebook/artifact、作者最終feature sourceを照合した。
- original Optuna group先頭5 bank、fixed-lag 192、公開tabular stackのv2手法契約と実行量を固定した。

- discussion stage 2-1 / 2-2と公開final Notebookを確認した。
- stage 2-2掲載scoreが5 PF + tabularでありPF単体scoreではないことを確認した。
- 公開履歴には現行final Notebookしかなく、stage 2-2 exact config/sourceが取得できないことを確認した。
- `pf_1 × twGR × fixed-lag192`、最終公開parameter、direct decodeというproxy範囲をユーザーへ説明した。
- ユーザーが2026-08-07に「それで進めてください」とproxy実装・late submitを明示承認した。
- `input / target / output / loss / decode / context unit`と再現性設計を固定した。
- steeringを元にexp517をexp516から作成した。
- compact self-contained inferenceを`pf_1 × twGR × fixed-lag192`へ変更した。
- anchor/emission/ML/他bank/他representation/full smootherが実行されないcontract testを作成した。
- dynamic hidden-test、sample alignment、SHA manifest、LATE SUBMIT表示を維持した。
- static、Ruff F821、contract test 7件、Jupytext pairing、strict validationをPASSした。
- Kaggle packageのmetadata/bootstrap/resource/quotaを監査し、T4 x2 version 1をpushした。
- Kaggle public commit runを完走し、outputを取得してsubmit-check FAIL/WARN 0を確認した。
- fixed version 1を`LATE SUBMIT` messageで1回だけ提出し、ref `55327703`を受付済み。
- ref `55327703`のhidden rerunを監視し、Public `7.825` / Private `9.689`、scoring 10分の`COMPLETE`を確認した。
- result、metrics、SESSION_NOTES、experiment summary、submission ledger、directionを更新した。
- negative resultが閉じるproxy tupleと検証不能なstage 2-2 claimを記録した。
