# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- なし。

## 完了

- `docs/06_reproducibility.md`を確認し、stable seed/SHA方針を設計へ記録した。
- exp237を親にexp248 scaffoldを作成した。
- configへcandidate bank、augmentation、variant、model、fold、guard、output contractを記録した。
- stable SHA256 keyによるbase row samplingとaugmentation割当を実装した。
- fixed shift / common datum / low-frequency drift / candidate・family・top dropout / spread scalingを実装した。
- viewごとのcandidate contextとmulti-observation score/MAE/NCC再計算を実装した。
- original-only / augmentedのbinary likelihood・expected-error GroupKFold学習を実装した。
- clean validationのAUC/logloss/Brier/calibration/topK/selected RMSE/fixed Viterbi評価を実装した。
- hidden-like、distance bucket、by-well、worst-well、margin calibration、augmentation inventory保存を実装した。
- Jupytext train notebookとtrain-side-only inference guard notebookを作成した。
- `SESSION_NOTES.md`、`result.md`、`metrics.json`、`experiment_summary.md`、`KAGGLE_DIRECTION.md`を更新した。
- Jupytext convert/test、py_compile、Ruff、strict validate-exp、validate-templateをpassした。
- Kaggle packageを作成し、metadataとbootstrap内configの整合を確認した。package確認後、ユーザー承認を受けてpushした。
- Kaggle CPU train version 1を完了し、metrics / summary / model manifest / by-well等の必要な小生成物だけ取得した。
- augmentation inventory、OOF prediction、feature schema、20 model、metrics、by-well、summary、kernel logのSHAを記録した。
- 全5 adoption guardと全5 fold RMSEが悪化したため、augmentationを不採用としてbranchを閉じた。
