# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering docs を作成した。
- `exp094_projection_only_on_exp073` から `exp096_projection_fadein_after_prefix` を作成した。
- row-wise projection beta fade-in を実装した。
- config grid を degree4/c2、degree5/c1.5、beta 0.50/0.75、fade 250-750/250-1000 に限定した。
- train/inference notebook の import と実験名を exp096 に更新した。
- 再現性設計を `design.md` に記入した。
- `.venv/bin/python -m py_compile` を通した。
- `make validate-exp EXP=exp096_projection_fadein_after_prefix` を通した。
- Kaggle train/inference package を strict mode で生成した。
- Kaggle train v1 を実行し、output を取得した。
- metrics/result/SESSION_NOTES/experiment_summary/KAGGLE_DIRECTION を結果で更新した。
- guard 通過を確認し、`inference.selected_variant` を `degree4_beta0.75_c2_fade250_750` に固定した。
- Kaggle inference v1 を実行し、output を取得した。
- submit-check を実行し、PASS を確認した。
- 提出 v1 の `Notebook Threw Exception` を調査し、public exp073 inference output 依存が hidden rerun 非互換であると診断した。
- exp073 base prediction を current test から再生成する hidden-compatible inference v2 を実装した。
- Kaggle inference v2 を実行し、output を取得した。
- inference v2 の submit-check を実行し、PASS を確認した。
- inference v2 を提出し、ref `53896594` / Public LB 8.651 を記録した。
- exp073 raw anchor は改善したが exp077 ML route anchor には届かないため、anchor 昇格なしで完了と判断した。
