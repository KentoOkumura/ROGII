# exp174_typewell_late_range_ml_posthoc_clip_audit

## 状態

- ルート: ml_model
- 状態: completed_train_side_rejected_no_submit
- CV: 8.501281182 baseline。発火 policy はすべて悪化。
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-03
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`

## 仮説

test 3 wells は last known `TVT_input` が typewell TVT range の後半にあり、train 全体の予測対象 row も後半に集中している。したがって `known_last_pct` が高い well で ML 予測の `pred_pct` が typewell 前半へ落ちる行だけを条件付き shrink / clip すれば、局所的な大外しを抑えられる可能性がある。

ただし train には前半例外があり target TVT は単調でないため、hard lower bound や `pred_pct >= known_last_pct` 制約は使わない。

## 変更点

- exp148 の保存済み OOF prediction を固定し、LightGBM は再学習しない。
- raw train/test の typewell range と last known `TVT_input` から `known_last_pct` を計算する。
- `pred_pct < lower_bound` かつ `known_last_pct >= threshold` の行だけ、`alpha` で lower bound TVT へ shrink する。
- fixed lower bound `0.55/0.60/0.65/0.70` と `known_last_pct - margin` の小 grid を比較する。
- baseline と上位 policy だけ OOF gzip を保存し、全 policy は metrics CSV に保存する。

## 検証方針

- Fold: なし。保存済み OOF prediction の no-training posthoc audit。
- Group: well 別 metrics を保存する。
- Stratification: near `000_050`、`1000_plus`、known-last pct bucket、pred/target pct bucket、changed row。
- Leakage Check: gate と lower bound は typewell range、last known `TVT_input`、predicted TVT だけで作る。true TVT は scoring / readout に限定する。

## 実行入口

- 学習 notebook: `exp174_typewell_late_range_ml_posthoc_clip_audit_train.ipynb`
- 推論 notebook: `exp174_typewell_late_range_ml_posthoc_clip_audit_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp174_typewell_late_range_ml_posthoc_clip_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp174-typewell-late-clip-train --title 'exp174 typewell late clip train' --run-on-push --strict"`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | baseline 8.501281182 |
| Public LB | - |
| Private LB | - |

## 所見

- Kaggle train v1 を完了。kernel: https://www.kaggle.com/code/kentookumura/exp174-typewell-late-clip-train
- LightGBM 学習はなし。posthoc variants のみ、LightGBM config 0、fold 0、booster 0、control 再学習なし。
- lower bound `0.55/0.60/0.65` は発火 0 行で baseline と同一。
- 発火する policy はすべて悪化。最小悪化は `fixed_lb0p7_klp0p75_a0p25` の RMSE 8.501891、baseline から +0.000609。
- 最大発火 policy `known_last_m0p05_klp0p75_a0p25` は 13,657 rows / 14 wells を変更し、RMSE 8.518425、+0.017144 悪化。
- inference port / submit はしない。

## 次

typewell late-range を ML posthoc clip として続けない。PF/Beam 側に使う場合も hard invalid / direct clip ではなく、candidate feature / penalty の診断に限定する。

## 2026-07-03 push メモ

初回 push は `kernel_sources` と長い slug の問題で `SaveKernel` 400 が続いたため、最終的に `exp148-train` のみを input source とし、同じ exp174 のまま短い `exp174-typewell-late-clip-train` / `exp174 typewell late clip train` で push した。v1 は完了済み。

## 表記

用語は `backlog/KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
