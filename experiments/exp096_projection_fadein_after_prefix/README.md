# exp096_projection_fadein_after_prefix

## 状態

- ルート: ml_model
- 状態: submitted_public_lb_recorded
- CV: 9.397537231
- Public LB: 8.651
- Private LB: -
- Submit ID: 53896594
- 作成日: 2026-06-21
- 親実験: `exp094_projection_only_on_exp073`

## 仮説

exp094 の global projection-only postprocess は exp073 OOF を改善したが、known prefix 直後の distance 0-50 ft と tail rank 0-99 を悪化させた。projection correction 自体は long-tail 側で有効な可能性があるため、prefix 直後は correction を 0 にし、距離とともに fade-in すれば near-prefix continuity を守りながら overall 改善を残せるか確認する。

## 変更点

- exp073 `gpu_repro_guard_dp_threads8` / `lgb_mean` OOF prediction を固定入力にする。
- raw horizontal well の `MD/Z/TVT_input` から prefix anchor と `md_since` を復元する。
- exp094 と同じ `U = pred_tvt + Z - (anchor_t0 + anchor_z0)` projection correction を計算する。
- correction 適用時に row-wise beta を使う。
  - `md_since <= 250`: beta 0
  - `250 < md_since < fade_end`: beta を線形 fade
  - `md_since >= fade_end`: selected beta
- 候補は `degree4/c2`、`degree5/c1.5`、beta `0.50/0.75`、fade window `250-750/250-1000` に限定する。
- LightGBM 再学習、特徴量追加、PF/Beam 再生成、public notebook blend は行わない。

## 検証方針

- Fold: exp073 と同じ `GroupKFold(n_splits=5, group=well)` 相当、および well-hash fold
- Group: `well`
- Leakage Check: projection fit は `pred_tvt`, `MD/Z`, prefix `TVT_input` のみを使い、`target_tvt` は scoring だけに使う
- Guard: overall RMSE、fold delta、distance 0-50 / 50-100 / 100-250、tail rank 0-99、tail length bucket、correction p95、path continuity

## 実行入口

- 学習 notebook: `exp096_projection_fadein_after_prefix_train.ipynb`
- 推論 notebook: `exp096_projection_fadein_after_prefix_inference.ipynb`
- Kaggle 準備: `make prepare-kaggle-notebooks EXP=exp096_projection_fadein_after_prefix EXTRA_ARGS="--notebook train --run-on-push --strict"`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は明示 debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| baseline exp073 OOF RMSE | 9.526374817 |
| best fade-in projection OOF RMSE | 9.397537231 |
| delta vs exp073 | -0.128837518 |
| delta vs exp094 global best | -0.001918793 |
| best variant | `degree4_beta0.75_c2_fade250_750` |
| guard | PASS |

Kaggle train v1:

- Kernel: `kentookumura/exp096-projection-fadein-after-prefix-train`
- URL: https://www.kaggle.com/code/kentookumura/exp096-projection-fadein-after-prefix-train
- Output: `/tmp/kaggle-output/exp096_projection_fadein_after_prefix/train_v1`
- Runtime: 1137.359 秒
- Rows / wells: 3,783,989 / 773

Inference v2:

- Kernel: `kentookumura/exp096-projection-fadein-after-prefix-inference`
- URL: https://www.kaggle.com/code/kentookumura/exp096-projection-fadein-after-prefix-inference
- Output: `/tmp/kaggle-output/exp096_projection_fadein_after_prefix/inference_v2`
- GPU / internet off
- exp073 base prediction は exp073 train artifact から current test 上で notebook 内再生成
- submission rows: 14,151
- submission SHA: `41b251c4ef29aa9daee62768890b406621310b7977164c24923a979e844dbaf5`
- submit-check: PASS
- prediction range: 11591.730469 - 12239.677734
- diff vs exp073 inference v2: RMSE 0.984292, p95 abs diff 1.953125

Submission:

- ref: `53896594`
- Public LB: 8.651
- exp073 raw anchor 8.780 より改善
- exp077 ML route submitted/postprocessed anchor 8.611 より悪化

## 所見

### 良かった点

- exp094 の問題だった distance 0-50 ft と tail rank 0-99 の悪化は 0.0 に抑えた。
- original fold / well-hash fold は全 fold で改善側だった。
- best は exp094 global best よりもわずかに良く、prefix protection を入れても overall gain を失わなかった。

### 悪かった点

- tail length 0-499 は +0.001918 RMSE とごく小さい悪化が残った。ただし guard threshold 0.02 内。
- OOF postprocess grid の結果なので、hidden test で同じ改善を保証しない。

## リスク / 注意

- fade-in でも true jump を平滑化する可能性は残る。
- test-side projection feature parity と submit-check は PASS。
- 提出 v1 は public exp073 inference output 依存により hidden rerun で `Notebook Threw Exception`。v2 は public output copy ではなく current test 上で exp073 base を再生成する hidden-compatible 構成。
- v2 の提出 ref `53896594` は Public LB 8.651。exp073 raw よりは良いが exp077 anchor には届かないため、ML route anchor には昇格しない。

## 次

1. exp096 は提出済みとして完了。
2. fade-in projection 単独では exp077 を更新しないため、後続は別仮説に回す。
