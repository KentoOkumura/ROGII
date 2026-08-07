# exp508 output contract

Stage A実装は完了したがKaggle CPU実行は未承認であり、以下の生成物はまだ存在しない。
別途run承認後にだけ`artifacts/`へ保存する。

| 生成物 | 内容 |
| --- | --- |
| `input_manifest.json` | source path、file SHA、schema、rows/wells、environment version |
| `row_order_manifest.json` | logical key、global row order、well order、per-well row order、fold SHA |
| `postprocess_contract_resolved.yaml` | 実行時に解決したSG/warmup契約。設計値からの差分は許可しない |
| `trajectory_postprocess_predictions.parquet` | key、fold、control、SG primary、report-only 2本。truth/error列は含めない |
| `prediction_manifest.json` | 各prediction content SHA、finite/parity/short-well check |
| `short_well_audit.csv` | well別row数、effective SG window、filter適用有無 |
| `primary_fold_metrics.csv` | control / primaryのfold別RMSEとdelta |
| `primary_scope_metrics.csv` | 固定3 MD scopeとhidden-like 2面のRMSE/delta |
| `primary_by_well_metrics.csv` | well別rows、control/primary RMSE、delta |
| `prediction_start_continuity.csv` | well最初のscore rowにおけるcontrol/primary/correction |
| `trajectory_smoothness_by_well.csv` | well別correction量とcontrol/primaryのsecond-difference RMS |
| `trajectory_diagnostics.json` | pooled correction分布とsecond-difference norm要約 |
| `primary_decision_freeze.json` | report-only score前に固定したprimary decisionとgate SHA |
| `report_only_metrics.json` | primary decision freeze後のwarmup 2本の非選択score |
| `promotion_gate.json` | 事前固定all-AND各条件と最終判定 |
| `reproducibility_manifest.json` | input、contract、prediction、metrics、gateのSHA |

`trajectory_postprocess_predictions.parquet`はtruth-free prediction freezeを証明する正規prediction
artifactとする。truth/errorを含むjoined working tableは保存しない。gzip生成物を追加する場合はraw
gzip SHAとdecompressed content SHAを分け、後者を主証拠にする。
