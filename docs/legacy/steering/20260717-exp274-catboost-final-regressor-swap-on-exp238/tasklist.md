# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- 公開 notebook 由来 `cb0` の設定と source SHA を固定した。
- 再現性設計を `design.md` に記入した。
- CatBoost train notebook、disabled inference notebook、config、実験記録を実装した。
- Jupytext round-trip、py_compile、Ruff、strict validate-exp を通した。
- canonical kernel id / title で Kaggle train package を作成した。
- package metadata の GPU / internet / kernel source と bootstrap 内 config byte parity を確認した。
- GPU cost 契約をユーザーに提示し、承認後に canonical T4 kernel version 1 を実行した。
- Kaggle train `COMPLETE`、5 models、fold / stress / guard / SHA を確認した。
- raw CatBoost と固定0.25 blendがparent RMSEを改善せず、全raw guard FAILを記録した。
- feature schema、fold matrix、model、OOF、manifest、summary SHA を記録した。inference / submit 不採用のため submission SHA は対象外とした。
- exp274 を negative result として閉じ、CatBoost parameter rescueを行わないと判断した。
- reference-only inference notebookを実装し、Jupytext / py_compile / Ruff / strict validationを通した。
- canonical Kaggle inference packageを作成し、metadata、29-file bootstrap manifest、config byte parity、kernel sourcesを監査した。
- repository tests `124 passed`を確認し、canonical inference kernel version 1をT4でpushした。
- canonical inference kernel version 1 `COMPLETE`、T4 425.779秒、14,151行、fallback 0を確認した。
- raw CatBoost / parent LightGBM / fixed0.25 blendの3出力を取得し、公式sampleに対するsubmit-check、ID順、SHA、float32 blend式を確認した。
- 初回確認でraw CatBoost code submission `ref=54793316` を特定した。当時はKaggle API 2.2.2/2.2.3が`PENDING`でPublic Score未反映だった。
- target refを新しい別submissionから分離して再照合し、`ref=54793316`の`COMPLETE` / Public LB 7.715をKaggle APIで確認した。
- exp257 7.718を-0.003更新するML submitted anchorとして記録し、train-side guard FAILと不採用判断は維持した。
