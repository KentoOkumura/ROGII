# 設計

## アプローチ

`exp092` の inference boundary である `run_saved_model_inference()` に opt-in の `hidden_assert_probe` を追加する。通常の inference はデフォルト無効のままにし、probe を有効化した code submission rerun だけ hidden test 条件を assertion で検査する。

assert は hidden の数値をログへ出す診断ではなく、pass/fail だけを見る LB probing 用の仕組みにする。失敗時の例外 message には failed check 名のみを入れ、hidden 行数、well 数、prediction 統計、閾値超過量は出さない。

## Assert 条件

デフォルト設定として、probe 有効時に hidden context で次を assert する。

- `non_visible_signature`: 入力が exposed visible test signature ではないこと。通常 kernel や sample rerun を hidden probe と誤認しないための確認。
- `sample_id_coverage`: `sample_submission.id` の全行に対して、生成した prediction が存在すること。`fallback_rows == 0` と `submission_rows == predicted_rows` を要求する。
- `finite_predictions`: `last_known_tvt`、`pred_delta`、`pred_tvt` がすべて finite であること。
- `anchor_t0_abs_max`: raw well CSV から復元した prefix anchor `TVT_input` と inference feature の `last_known_tvt` の最大差が `0.05` 以下であること。
- `known_prefix_rows_min`: 各 hidden well に少なくとも 1 行の既知 `TVT_input` prefix があること。
- `well_step_abs_p95_max`: well 内で `pred_tvt` を id suffix 順に並べた隣接差分の p95 が全 well で `2.0` 以下であること。
- `well_step_abs_max_max`: well 内で `pred_tvt` を id suffix 順に並べた隣接差分の最大値が全 well で `10.0` 以下であること。
- `pred_delta_abs_p95_max`: well 内の `|pred_tvt - last_known_tvt|` p95 が `100.0` 以下であること。
- `pred_delta_abs_max_max`: well 内の `|pred_tvt - last_known_tvt|` 最大値が `160.0` 以下であること。
- `pred_range_max`: well 内の `pred_tvt` range が `180.0` 以下であること。
- `near_prefix_delta_abs_p95_max`: 各 well の先頭 `250` prediction rows で `|pred_delta|` p95 が `25.0` 以下であること。
- `near_prefix_delta_abs_max_max`: 各 well の先頭 `250` prediction rows で `|pred_delta|` 最大値が `50.0` 以下であること。
- `near_prefix_step_abs_p95_max`: 各 well の先頭 `250` prediction rows で `pred_tvt` 隣接差分 p95 が `1.5` 以下であること。
- `near_prefix_step_abs_max_max`: 各 well の先頭 `250` prediction rows で `pred_tvt` 隣接差分最大値が `5.0` 以下であること。
- `projection_feature_finite`: U-projection / disagreement feature がすべて finite であること。
- `projection_correction_abs_p95_max`: projection correction / residual 系 feature の列別 abs p95 が `20.0` 以下であること。
- `projection_correction_abs_max_max`: projection correction / residual 系 feature の列別 abs max が `80.0` 以下であること。
- `u_disagreement_abs_p95_max`: PF/Beam/likelihood-PF U-space disagreement 系 feature の列別 abs p95 が `250.0` 以下であること。
- `u_disagreement_abs_max_max`: PF/Beam/likelihood-PF U-space disagreement 系 feature の列別 abs max が `500.0` 以下であること。

これらの追加条件は、exp092 の train-side 懸念である by-well regression、near-row inconclusive、projection/disagreement 過補正を、hidden の正解なしで観測できる proxy に落としたもの。

`skip_visible_test: true` の場合、通常 notebook の exposed visible test signature では hidden assert を skip し、実装確認だけ通す。

## 実験範囲

- 対象実験: `exp092_u_projection_correction_disagreement_fullrun`
- Route: `ml_model`
- 親実験: `exp085_u_projection_feature_ablation`
- 変更する変数: inference notebook / saved-model inference helper / config / records
- 固定する変数: exp092 train artifact、selected variant `u_projection_correction_plus_disagreement`、selected model `lgb1`、既存 submitted anchor `ref=53927479`

## 再現性設計

- seed policy: 新規 RNG なし。既存 exp092 inference の deterministic PF/Beam replay と saved LightGBM booster inference を使う。
- stochastic 処理の有無: probe 自体にはなし。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成ロジックなし。既存 inference flow が再生成する target-free features のみ。
- 並列処理と乱数の関係: assertion は pandas / numpy の deterministic 集計のみ。
- CPU/GPU runtime と deterministic flags: 既存 inference 設定を変更しない。
- train cache / test feature regeneration の SHA 記録方針: normal inference output は従来通り prediction SHA / submission SHA を記録する。hidden probe の pass/fail は submission ref と実行 version に紐づけて記録し、hidden 集計値は記録しない。
- model manifest / prediction / submission SHA 記録方針: visible run では従来通り。hidden submission rerun は Kaggle 側の pass/fail を正とし、hidden details はログに出さない。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks.py --notebook inference --strict` で config と notebook の整合を確認する。

## リスク

- リークリスク: hidden 数値をログや例外に出さず、pass/fail だけを信号にする。
- CV/LB 不一致リスク: assertion は性能改善を保証しない。落ちた場合は hidden assumption が崩れたことだけを示す。
- ランタイム/メモリリスク: prediction 生成後の軽量 groupby のみ。
- 再現性リスク: code submission rerun の hidden 入力はローカル再現できないため、submission ref / kernel version / probe config を記録する。
