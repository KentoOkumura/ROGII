# exp094_projection_only_on_exp073

## 状態

- ルート: ml_model
- 状態: completed_no_inference_guard_failed
- CV: 9.399456024
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-20
- 親実験: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`

## 仮説

exp073 full replay の raw `pred_tvt` は `TVT + Z - prefix_anchor` 空間では局所的に滑らかな軌跡を持つ可能性がある。再学習せず、well 内の target-free polynomial projection だけを後処理として足すことで、近距離 row や short tail を壊さずに OOF RMSE を改善できるかを確認する。

## 変更点

- exp073 `gpu_repro_guard_dp_threads8` / `lgb_mean` OOF prediction を固定入力にする。
- raw horizontal well の `MD/Z/TVT_input` から prefix anchor と `md_since` を復元する。
- `U = pred_tvt + Z - (anchor_t0 + anchor_z0)` を well ごとに robust polynomial fit し、degree / beta / robust C grid を比較する。
- LightGBM 再学習、特徴量追加、PF/Beam 再生成、public notebook blend は行わない。

## 検証方針

- Fold: exp073 と同じ `GroupKFold(n_splits=5, group=well)` 相当、および well-hash fold
- Group: `well`
- Stratification: なし
- Leakage Check: projection fit は `pred_tvt`, `MD/Z`, prefix `TVT_input` のみを使い、`target_tvt` は scoring だけに使う
- Guard: overall RMSE、fold delta、distance bucket、tail rank bucket、tail length bucket、near-row regression、correction p95

## 実行入口

- 学習 notebook: `exp094_projection_only_on_exp073_train.ipynb`
- 推論 notebook: `exp094_projection_only_on_exp073_inference.ipynb`
- Kaggle 準備: `make prepare-kaggle-notebooks EXP=exp094_projection_only_on_exp073 EXTRA_ARGS="--notebook train --run-on-push --strict"`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は明示 debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | 9.399456024 |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- 実装は postprocess-only なので、exp073 anchor との差分を projection の効果として切り分けられる。
- best `degree4_beta0.75_c2` は exp073 OOF RMSE 9.526374817 から 9.399456024 へ改善した。

### 悪かった点

- 同一 OOF 上の grid 比較なので、全体 RMSE 改善だけでは採用できない。
- best は near-row を壊し、distance 0-50 ft の RMSE delta が +1.439466、tail rank 0-99 の RMSE delta が +1.130416 だった。
- 全 grid variant で near-row guard を通過できなかった。

### リスク / 注意

- projection は真の急変や prefix 直後の continuity を平滑化する可能性がある。今回 near-row guard failed のため inference port しない。
- inference は `inference.selected_variant` が null の間は submission を作らない。

## 次

1. exp094 は inference port せず完了扱いにする。
2. projection を続けるなら global postprocess ではなく near-prefix を除外する gated projection / long-tail-only projection として別実験化する。
3. 現時点の優先は exp092 の U-projection correction+disagreement fullrun と PF candidate ranker 側に置く。

## 表記

用語は `backlog/KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
