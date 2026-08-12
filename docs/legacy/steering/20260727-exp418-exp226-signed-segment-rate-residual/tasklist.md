# タスクリスト

## 目的

exp333のoffset targetだけをsigned cumulative-rate targetへ置換する一因子実験を、
design-only状態から承認単位ごとにfail closedで進める。

## 固定した比較

- 親: exp333
- base prediction: exp226
- 変更: targetとcorrection applicationのみ
- 固定: K16、fold、136特徴、LightGBM 1 config、評価scope

## 承認待ち・未着手

- なし。Stage 0 technical FAILによりexp418の後続branchは閉鎖した。

## ブロック中

- Stage 1、推論、提出:
  Stage 0が`FAIL_CLOSE_BRANCH`のため停止。現行exp418内では再開しない。

## 完了

- exp418を次の未使用実験番号として確定した。
- steering `requirements.md`、`design.md`、`tasklist.md`を作成した。
- target、符号、単位、K16 basis、integration、fold、feature、model、gateを固定した。
- 実験scaffoldを作成した。
- `config.yaml`、README、SESSION_NOTES、result、metricsをdesign-onlyへ更新した。
- `KAGGLE_DIRECTION.md`のbacklogへ追加し、P3高リスクCPU案として優先度を固定した。
- `experiment_summary.md`へlineageとdesign-only状態を追加した。
- 2026-07-27の実装承認を記録した。
- exp333 Stage 1 SHA manifestからnested prediction、fold manifest、feature schemaを
  fail-closedで解決するloaderを実装した。
- exp226 fit/regenerationなしで、exp333-compatible 136-feature再構築と
  feature-freeze SHA検証を実装した。
- zero-intercept K16 rate basis、float64 lstsq target、continuous integration、
  Stage 0 oracle、Stage 1 5-fold LightGBM、scope/tail/rate gate、生成物/SHA保存を
  compact self-contained train候補へ実装した。
- Jupytext候補`.ipynb`を生成し、round-trip、py_compile、Ruff F821をPASSした。
- 専用test 14件を追加し、全件PASSした。
- 2026-07-28のユーザー依頼で、正規train Notebook採用、Kaggle package、
  Stage 0 push/runの承認を得た。
- 正規train Notebookを採用し、strict packageのmetadata / bootstrap / config
  整合を確認した。
- Kaggle private CPU version 1（id_no `128832515`）でStage 0を実行した。
- 3,783,989 rows / 773 wells / 12,368 segments、0 model / 0 booster /
  0 exp226 fitを確認した。
- rate oracleはRMSE `0.6469514161595739`、exp226比
  `+8.780158180422646 ft`、5/5 folds改善だった。
- matrix / sequential integration差`6.295408638834488e-12 ft`が固定上限
  `1e-12 ft`を超えたためtechnical 8/9 PASS、`FAIL_CLOSE_BRANCH`となった。
- summary / rate-targetのfile / content SHAをconfig、metrics、result、
  SESSION_NOTESへ記録した。
- Stage 1 approvalを有効化せず、exp418をterminal fail closedとした。
