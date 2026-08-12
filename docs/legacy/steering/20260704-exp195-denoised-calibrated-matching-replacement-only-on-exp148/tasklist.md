# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- exp195 steering 作成。
- exp190 の DCM generator / train skeleton を exp195 用にコピー。
- helper / train / inference script 名を exp195 に合わせた。
- active variant を `denoised_calibrated_matching_replacement_only` に変更。
- active model feature group から `learned_likelihood_confidence` を外し、`projection_correction + u_disagreement + denoised_calibrated_matching` にした。
- control 再学習なし、単一 GPU notebook、15 boosters 予定を `SESSION_NOTES.md` に記録した。
- Jupytext conversion / `--test`、py_compile、ruff F821/F401、experiment validation を完了した。
- train Kaggle package を `kentookumura/exp195-dcm-replace-exp148-train` / title `exp195 dcm replace exp148 train` で strict prepare した。
- Kaggle train v1 を実行し、logs / metrics / SHA を `SESSION_NOTES.md` と `result.md` に記録した。
- train-side CV が exp148 / exp190 から大きく悪化したため、current-test feature generation / inference port / submit を行わない判断にした。
