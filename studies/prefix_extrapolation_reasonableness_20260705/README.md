# Prefix extrapolation audit inputs and outputs

このディレクトリには、known prefixからの直接外挿を比較した生の集計表とsummary JSONを置きます。人間向けの結果、使用制約、判断は、[Prefix Extrapolation Reasonableness Audit](../../docs/surveys/prefix_extrapolation_reasonableness_20260705.md)を正とします。

## 実行条件

- 入力: `data/raw/train`の773 usable wells
- fit範囲: `TVT_input`が既知のprefix rowsだけ
- 評価範囲: train `TVT`を参照できるhidden/evaluation rows
- 性質: submission候補ではなくpre-experiment diagnostic

## 再実行

```bash
uv run python studies/prefix_extrapolation_reasonableness.py
```

## 生の出力

`overall_metrics.csv`、`well_summary.csv`、`by_well_metrics.csv`、`step_bucket_metrics.csv`、`drift_bucket_metrics.csv`、`summary.json`を保存します。集計の解釈はsurveyへ集約します。
