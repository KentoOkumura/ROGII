# exp226 offset root-cause audit inputs and outputs

このディレクトリには、保存済みgroup-safe exp226 OOF 3,783,989行・773 wellsを読む監査の生の表とsummary JSONを置きます。人間向けの結論、証拠の限界、後続実験への制約は、[exp226 オフセット根本原因監査](../../docs/surveys/exp226_offset_root_cause_audit_20260727.md)を正とします。

新規学習、推論、提出は行いません。

## 再実行

```bash
uv run python studies/exp226_offset_root_cause_audit.py
```

## 生の出力

`stage_metrics.csv`、`segment_persistence_metrics.csv`、`persistent_offset_episodes.csv`、`well_root_cause_readout.csv`などの表と`summary.json`を保存します。各数値の解釈はsurveyへ集約します。
