# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし。

## 完了

- 仮説を exp291 として採番した。
- safe base絶対保持、self-GR完全除外、Type Well GR local modes全件保持を固定した。
- candidate、checkpoint、commit、truth freeze、guard、no-rescue条件を固定した。
- steeringを作成した。
- 実験ディレクトリと計画用config/README/session/result/metricsを作成した。
- `KAGGLE_DIRECTION.md` と `experiment_summary.md` にexp291を記録した。
- experiment validatorとproject strict config validationを通過した。
- ユーザーの実装承認を受けた。
- 別名compact self-contained train/inference sourceとnotebookを実装した。
- local-mode全保持、safe保持、matched-count、persistent commit、truth freezeを実装した。
- exp291 dedicated 9 tests、Ruff、py_compile、Jupytext round-trip、strict experiment validation、
  repository 260 testsを通過した。
- 親exp284 compactと同じ10章を維持し、2427行に対してexp291 trainは2365行であることを確認した。
- ユーザーからcanonical train採用とKaggle CPU 1回実行の承認を得た。
- strict Kaggle packageのmetadata、bootstrap config、source、SHAを監査した。
- 固定計数を再確認し、canonical CPU kernel version 1を1回実行した。
- 766 eligible wells / 5 foldsでtechnical guardが全PASSしたことを確認した。
- 性能・安全性guard FAILを確認し、parameter rescueなしでbranchを閉じた。
- `config.yaml`、`metrics.json`、`SESSION_NOTES.md`、`result.md`、`README.md`、
  `KAGGLE_DIRECTION.md`、`experiment_summary.md`へ結果を反映した。
- decoder、推論、prediction、submissionを生成していないことを確認した。
