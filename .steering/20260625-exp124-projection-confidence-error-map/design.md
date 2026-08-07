# 設計

## 背景

`exp094_projection_only_on_exp073` は exp073 OOF `lgb_mean` に対して projection-only postprocess を適用し、RMSE を `9.526374817 -> 9.399456024` へ改善した。一方で distance `0-50 ft` と tail rank `0-99` が大きく悪化し、global policy としては guard failed だった。

`exp096_projection_fadein_after_prefix` は near-prefix を beta 0 にする実用 follow-up だが、`projection_confidence_error_map` は別目的で、projection がどの well 条件で効くかを読む診断に限定する。

## 入力

- 必須:
  - exp094 best predictions: `exp094_projection_only_on_exp073_best_predictions.csv.gz`
  - raw train horizontal wells: `data/raw/train/*__horizontal_well.csv`
- 任意:
  - `artifacts/pf_beam_disagreement_error_map/pf_beam_disagreement_well_map.csv`
  - exp083 well summary fallback
  - exp065 `common_typewell_cluster_assignments.csv`

## 処理

1. exp094 best predictions から `base_pred_tvt`、`pred_tvt`、`target_tvt`、`md_since`、`tail_rank`、`tail_length`、`projection_correction_applied` を読む。
2. raw train を well ごとに読み、scored row の `Z`、`GR`、prefix 長、eval Z span、eval GR missing rate、prefix GR missing rate を復元する。
3. optional context があれば well 単位で結合する。
4. row 単位で raw/base error、projected error、absolute error delta、squared error delta を作る。
5. 以下で bucket metrics を作る。
   - distance bucket
   - tail rank bucket
   - tail length bucket
   - prefix length bucket
   - Z span quantile bucket
   - GR missing quantile bucket
   - PF/Beam disagreement bucket
   - native typewell group
   - selected cross buckets
6. gate candidate metrics を作る。
   - baseline
   - global projection
   - longtail only
   - tail length gate
   - high Z span gate
   - longtail + high PF/Beam disagreement
   - longtail + native group size

## 出力

- `exp124_projection_confidence_error_map_row_error_map.csv.gz`
- `exp124_projection_confidence_error_map_well_error_map.csv`
- `exp124_projection_confidence_error_map_bucket_metrics.csv`
- `exp124_projection_confidence_error_map_gate_metrics.csv`
- `exp124_projection_confidence_error_map_summary.json`
- `README.md`

## 判定

この実験単体では submit しない。次へ進む条件は、raw-test-compatible な条件で near rows を壊さず、longtail または high-disagreement bucket に一貫した改善が見えること。条件が弱い場合は projection gate は exp096 の fade-in 以上に進めない。
