# タスクリスト

## 完了

- 依頼原文とKaggle Discussion `733319`から入力、出力、尤度、推論方法、処理単位を記録した。
- 元コードと一致しない近似実装であるため、実装区分を`proxy`とした。
- Jupytext percent形式でself-contained train / inferenceを実装した。
- forward-backwardを小型の総当たり計算と照合した。
- 検証坑井を兄弟井の参照波形作成から除外する処理と専用testを実装した。
- py_compile、Ruff、Jupytext、strict experiment validationを通した。
- Kaggle CPU train version 1で773坑井、3,783,989行のOOFを完了した。
- 全体RMSE `40.88961598374063`、fold別・3種類別RMSE、実行時間、予測内容SHA256を記録した。
- 公開OOF `5.9703`を再現できなかったことを記録した。
- ユーザー指示に従い、開始直後の推論Notebookを削除し、推論結果を使用しなかった。
- late submissionを行わず、提出回数0として記録した。
- 今回の近似実装だけを閉じ、3位チームのHMM全体を否定しない範囲を`result.md`へ記録した。
- ユーザー判断により、再現未達・推論中止・未提出の結果で実験を完了した。

## 実施対象から外した項目

- hidden testの推論結果生成。
- submission.csvの生成と提出前検査。
- `LATE SUBMIT`の実施とLB取得。

上記は、再現できていない場合は推論へ進まないというユーザー指示によって実施対象から外した。
