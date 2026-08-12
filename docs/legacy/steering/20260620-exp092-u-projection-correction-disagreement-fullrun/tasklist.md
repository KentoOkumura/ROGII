# タスクリスト

## TODO

- by-well max regression warning に対する worst-well gating / regression guard を検討する。
- raw-test projection feature parity を監査する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- `docs/legacy/steering/20260620-exp092-u-projection-correction-disagreement-fullrun/` を作成した。
- `experiments/exp092_u_projection_correction_disagreement_fullrun/` を exp085 から作成した。
- `config.yaml` を single-variant fullrun 用に更新した。
- `u_projection_correction_disagreement_fullrun.py` に出力 prefix / experiment 名を反映した。
- train / inference notebook を exp092 用に更新した。
- `README.md`、`SESSION_NOTES.md`、`result.md`、`metrics.json` を exp092 用に更新した。
- 静的検証を通した。
- synthetic smoke test で fullrun runner の最低限の挙動を確認した。
- Kaggle train package を作成し、metadata と bootstrap 内 config/helper SHA の整合を確認した。
- Kaggle train v1 を実行して正式 pooled OOF / bucket / importance / prediction SHA を取得した。
- output 取得後に feature content SHA、prediction SHA、model SHA、Kaggle kernel version を記録した。
- exp073 / exp077 OOF と exp092 predictions を align し、by-well delta、near-row / long-tail、path continuity を guard した。
- ユーザー依頼により inference v1 を作成し、Kaggle で実行した。
- inference output の submit-check を通した。
- ユーザー手動提出により `ref=53927479` / Public LB 8.350 を記録した。
