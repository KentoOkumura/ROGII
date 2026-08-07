# 要件

## 依頼

- exp306 Stage 0で唯一full-eligibleになったL1だけを、全773 wellsで監査する候補として`exp351_exp306_l1_full_convergence_audit`に切り出す。
- backlog、実験ディレクトリ、steeringを作成し、実装前の設計を確定する。
- 今回はdesign-onlyとし、solver実装、正規Notebook実装、Kaggle package/push/run、科学評価、inference、submissionは行わない。

## 仮説

exp306 Stage 0で固定L1設定が128/128 seriesの収束、finite/order/fallback、runtime、exact parityを通過したため、設定を変えずに全773 wells / 1,546 seriesへ展開しても全件technical PASSできる。

## 親証拠

- 親: `exp306_robust_rts_l1_convergence_calibration_audit`。
- Kaggle kernel: `kentookumura/exp306-rts-l1-convergence-calibration-audit-train` version 1、id_no `128231380`。
- L1 Stage 0: `128/128` convergence/technical PASS、iterations min/mean/max `264/656.758/1993`。
- L1 Stage 0 runtime: `25.16088893899996 sec`、773-well外挿`303.8963617163589 sec`。
- 8 wells x 2 series parity: output/status/iteration SHA完全一致。
- raw well identity content SHA: `bbb687a1998092578583ce259309b49031d095bde57cbb26c0ab8808d2379b32`。
- truth/scientific score、prediction、submissionは親Stage 0でも未生成。

## 制約

- Routeは`pf_beam`とする。
- 対象branchは`l1_iter2000_rho1_tol1e4`の1つだけとし、RTS A/Bを実装・実行しない。
- L1 objective、lambda式、ADMM `rho=1.0`、maximum iterations `2000`、absolute/relative tolerance `1e-4`、adaptive rho falseをexp306から完全固定する。
- exp306と同じraw input、missing GR policy、series coordinate、within-series orderを使う。
- horizontal CSVは`MD/GR/TVT_input`だけを読み、`TVT/error/abs_error/formation`をload時点で拒否する。
- typewell `TVT/GR`は参照seriesの座標・信号にだけ使う。
- truth、error、formation、MRR、top3、RMSE、hidden-like role、exp304 scientific scoreを読まない。
- 773 wells x horizontal/typewell = 1,546 seriesを1回だけsolver実行する。full solver rerunや設定gridは行わない。
- model / LightGBM config / trained fold / HMM / PF / Beam / booster / 親control再実行 / GPUはすべて0とする。
- CPU、internet off、single worker、BLAS threads 1を固定する。

## 受け入れ基準

- 親version 1のscientific contract、Stage 0 gate、sample manifest、summaryのfile/content SHAが事前固定値と一致する。
- raw trainのwell集合が773 wellsで、raw well identity content SHAが親の`bbb687a1...b32`と一致する。
- statusが773 wells x 2 series = 1,546 rowsで、well/series coverageに欠損・重複がない。
- `1,546/1,546` seriesがconvergedかつtechnical PASSする。
- 全seriesでfinite input/output、length/order identity、silent fallback 0、error空を満たす。
- exp306固定64-well sampleをfull出力から抽出し、prepared input、L1 output、L1 status content SHAが親Stage 0と完全一致する。
- exp306 parity先頭8 wellsについてoutput/status/iteration content SHAが親parity manifestと完全一致する。
- input/output/statusをcanonical順で保存し、content SHA、raw gzip SHA、decompressed SHA、row countを記録する。
- full audit全体のwall timeが`30,600 sec`（8.5時間）以内である。
- すべてのAND gateを通過した場合だけ、将来の別科学評価へ進めるtechnical eligibilityを得る。
- 1 seriesでもFAIL、parent SHA不一致、cross-run parity不一致、runtime超過があればtechnical negativeとして閉じ、iteration/tolerance/lambda/rho/gridで救済しない。

## 承認範囲

- 2026-07-23の依頼はdesign-only scaffold、steering、backlog、実験記録の作成を承認する。
- 2026-07-23の追加依頼`exp351を実装してください`により、compact self-contained train/inference候補、synthetic contract tests、config/実験記録更新を承認した。
- 2026-07-23の追加依頼`実行してください`により、正規Notebook採用とKaggle CPU full auditのpackage/push/run 1回を承認した。
- technical PASS後の科学評価、denoiser選択変更、HMM/PF/Beam、inference、submissionはさらに別実験・別承認とする。

## 次のアクション

Kaggle CPU version 1は`1,537/1,546` convergenceでall-series gateをFAILした。technical negativeとして閉じ、iteration/tolerance/lambda/rho/grid救済、scientific score、inference、submissionを行わない。
