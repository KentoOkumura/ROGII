# 要件

## 依頼

`typewell_late_range_continuity_selector_on_exp176` backlog を実験化する。exp176 の typewell late-range prior 入り candidate error surface は row-wise OOF RMSE が良い一方、path switch が高いため、exp158-style Viterbi / segment continuity selector で平滑化する。

## 制約

- Route: `ensemble`
- 親実験: `exp176_typewell_late_range_pfbeam_candidate_prior`
- continuity 参照: `exp158_segment_continuity_selector_on_exp157`
- 新規 LightGBM 学習はしない。exp176 v3 の saved boosters 15 本を読む posthoc audit に限定する。
- `selected_tvt` を direct replacement、blend、postprocess、submit に使わない。
- candidate_pct threshold は exp176 と同じ target-free 設定に固定する。
- valid/test true TVT、oracle best、true-error rank は selector feature や Viterbi selection に使わない。
- 再現性は `docs/06_reproducibility.md` に従い、入力 cache、exp176 model manifest、OOF prediction の SHA を記録する。

## 受け入れ基準

- `experiments/exp191_typewell_late_range_continuity_selector_on_exp176/` に config、helper、train/inference notebook 起点 `.py`、notebook、記録ファイルがある。
- exp176 feature schema と model manifest を解決し、exp176 v3 と同じ `tlp_` / `candidate_tlp_` feature contract で score surface を復元する。
- Viterbi grid、比較対象、出力生成物が config と notebook 上で確認できる。
- Kaggle push 前コストとして、新規 booster 0、parent/control 再学習なし、saved booster inference 15 本を `SESSION_NOTES.md` に記録する。
- 静的検証、Jupytext 変換、experiment validation が通る。
