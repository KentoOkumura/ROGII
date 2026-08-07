# exp506 output contract

Stage Aが保存する生成物を固定する。version 2で全12生成物を生成し、
`kaggle/output/stage_a_v2/artifacts/`へ取得してfile SHAをreproducibility manifestと照合済み。

| 生成物 | 内容 |
| --- | --- |
| `anchor_resolution_manifest.json` | exp497 terminal decision、selected anchor、source file/column/CV/SHA |
| `input_manifest.json` | exp413/exp497、exp490、fold、hidden-likeのfile/SHA/rows/wells/schema |
| `correction_manifest.json` | `exp490-exp357` logical content SHA、finite/key/parity checks |
| `meta_fold_weights.csv` | held fold、fit folds、lambda、bound hit、fit-row count |
| `primary_oof_predictions.parquet` | key、fold、anchor、correction、lambda、primary prediction |
| `primary_fold_metrics.csv` | anchor / primary RMSEとdelta、fold 0--4 |
| `primary_scope_metrics.csv` | 固定5 scopeのanchor / primary RMSEとdelta |
| `primary_by_well.csv` | anchor / primary RMSEとdelta、well単位 |
| `primary_gate.json` | technical、leakage、pooled、fold、scope、tail、weight stabilityの全AND判定 |
| `report_only_control.json` | primary freeze後のconvex control weight/RMSEとresidual相関診断 |
| `metrics.json` | primary CV、fold、scope、tail、weight、decisionの機械可読要約 |
| `reproducibility_manifest.json` | 全生成物SHA、kernel version、runtime、package/config SHA |

Primary decisionと`primary_gate.json`をfreezeする前に、report-only controlを評価してはならない。
FAIL時はinference artifact、current-test prediction、`submission.csv`を生成しない。
