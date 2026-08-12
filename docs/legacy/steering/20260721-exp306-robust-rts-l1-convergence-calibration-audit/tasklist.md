# タスクリスト

## TODO

- RTSを再訪する場合は、残る20 Stage 0 FAILのtarget-free failure profileと単一変更を別steeringで事前固定する。exp306内では追加gridを行わない。

## 進行中

- なし

## ブロック中

- L1 full auditは後続exp351へdesign-onlyで移管済み。exp351の実装・Kaggle実行は未承認。

## 完了

- backlog候補を`exp306_robust_rts_l1_convergence_calibration_audit`として採番した。
- Stage 0の固定64 wells、RTS A→Bの単一条件分岐、L1 max2000、8-well parity、branch別full auditを固定した。
- truth/scientific scoreを使わないtechnical-only境界と、exp304 selected SWTを変更しない契約を固定した。
- 最大実行量とmodel/LightGBM/HMM/PF/Beam/booster 0を記録した。
- `docs/06_reproducibility.md`に沿ってsample/input/output/status SHA、thread/runtime、Kaggle bootstrapを設計した。
- compact self-contained trainとfail-closed inferenceを実装し、正規Notebookへ採用した。
- exp304 target-free common preparation、robust RTS、L1 trendの必要関数だけを抽出し、親compactとの章立て・行数を比較した。
- horizontal `TVT`、truth/error/formation/score列を拒否するschema guardを実装した。
- RTS A→B条件分岐、L1 fixed max2000、64-well SHA sample、sample順先頭8-well parity、runtime projection、branch gateのsynthetic testを追加した。
- Stage 0最大384 core series-runs、parity最大32、full最大2 branches x 1,546 series-runs、model/LightGBM/HMM/PF/Beam/booster 0を再集計した。
- Jupytext test、構文、Ruff、専用test、`make validate-exp`、`make validate-template`をPASSした。
- push前にmetadataとbootstrap内config、stage、solver settings、approval flagを照合した。
- Kaggle CPU Stage 0 version 1（id_no `128231380`）を実行し、384 core + 16 parity series-runsを完了した。
- L1は128/128 convergence、exact parity、runtime、SHAをPASSし、唯一のfull-eligible branchになった。
- RTS A/Bは7/128・108/128 convergenceで不適格とし、事前契約どおり追加救済を行わず閉じた。
- input/output/statusのraw/decompressed SHAを取得ファイルから再計算し、gate記録値との一致を確認した。
- full audit、科学評価、実データinference、submissionは行っていない。
- L1 full audit候補を`exp351_exp306_l1_full_convergence_audit`として別steering/scaffold/backlogへ切り出し、exp306をStage 0 evidence anchorとして固定した。
