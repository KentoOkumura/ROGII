# exp497 出力契約

Stage別生成物を以下に固定する。Stage 0およびStage P/M/Eの実行コードは実装済みで、
ユーザーのKaggle実行承認を得た。実行生成物は各Kaggle kernel完了後にSHA付きで記録する。

## Stage 0

- `stage_0_source_audit.json`: source SHA、必要symbol、除外marker count
- `stage_0_input_fold_row_manifest.json`: exp413 OOF/fold/scope/hidden/by-well SHA、row/fold契約
- `stage_0_inner_fold_manifest.csv`: outer holdoutごとのinner 4 GroupKFold well割当
- `stage_0_spatial_pool_ledger.csv`: outer-valid除外済みspatial pool台帳
- `stage_0_execution_inventory.json`: PF/Beam/NCC/booster/Ridge実行量
- `stage_0_feature_schema_plan.json`: SP45 195列、learned 205列、float32 memory見積り
- `stage_0_preflight.json`: 上記SHAと0-model/0-path停止証拠

## Stage P

- `stage_p_fold{0..4}_physical_features.parquet`: target-free physical candidate partition
- `stage_p_fold{0..4}_well_metadata.parquet`: well形状とPF seed/sigma metadata
- `stage_p_fold{0..4}_feature_schema.csv`: 195/205/213列schema
- `stage_p_fold{0..4}_summary.json`, `stage_p_summary.json`: PF/Beam実行量とSHA ledger

## Stage M1/M2

- `stage_m_outer{0..4}_predictions.parquet`: truth attach前にfreezeしたouter-valid component OOF
- `stage_m_outer{0..4}_model_manifest.json`: LGB 24 + Cat 16、Ridge 2のmanifest
- `stage_m_outer{0..4}_weights.json`, `selector_policies.json`
- `stage_m_outer{0..4}_feature_importance.parquet`
- `stage_m_outer{0..4}_spatial_audit.json`, `summary.json`

## Stage E

- `component_oof.parquet`: SP45 residual、physical selector、projected SP45、learned trajectory、strict public-core、exp413、cross-fit blend
- `meta_fold_weights.csv`
- `fold_metrics.csv`, `scope_metrics.csv`, `hidden_like_metrics.csv`, `by_well_metrics.csv`
- `promotion_gate.json`, `reproducibility_manifest.json`, `stage_e_summary.json`

## 停止条件

promotion gate FAIL時はselected predictionをexp413に固定し、current-test prediction、
submission.csv、inference packageを生成しない。PASS時も推論実装とKaggle実行は
別承認とし、同じexp497内で継続する。
