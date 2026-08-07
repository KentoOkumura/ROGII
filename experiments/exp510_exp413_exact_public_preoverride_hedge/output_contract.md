# exp510 output contract

候補notebookはKaggle version 4で実行済み。Kaggle Notebook outputでは次を保存する。

- `sp45_projection_submission.csv`: raw hidden testからstable seedで再生成したprojected-SP45。
- `submission_B.csv`: SHA固定した3 boosterを適用したPipeline-B予測。
- `public_preoverride_submission.csv`: `0.55 * SP45 + 0.45 * Pipeline B`。
- `submission.csv`: `0.90 * exp413 + 0.10 * public_preoverride`。外部提出は別承認。
- `exp510_component_readout.csv.gz`: `id`、`well`、MD horizon、4 componentとfinal。
- `exp510_component_readout.json`: component間とstart continuityのtruth-free差分量。
- `exp510_by_well_readout.csv`: well別のRMSE/MAE/p95/max差分。
- `exp510_horizon_readout.csv`: 事前固定MD horizon別の差分。
- `exp510_reproducibility_manifest.json`: source/dataset/model/input/feature/prediction/submission SHA、seed policy、formula parity、fallback/duplicate/nonfinite件数。

exp413は公開test固定gzipを入力せず、dynamic sample上で再生成する。生成gzipのraw/decompressed SHAを
記録し、CSVを読み戻したserialized component boundaryをblendへ使う。public output CSVも入力にせず、
raw hidden testからSP45/Pipeline-B componentを生成する。モデル候補の欠落・複数候補・SHA不一致、sample
ID不一致、exp413 serialization drift `>1e-3`、fallback、NaN/Inf、formula parity `>1e-12`でfail-closeする。
