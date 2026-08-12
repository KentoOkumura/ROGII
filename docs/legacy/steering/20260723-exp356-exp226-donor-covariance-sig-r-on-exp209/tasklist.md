# タスクリスト

## TODO（block解除・別途承認後）

- Stage 0 donor-covariance generator、compact train候補、contract testsを実装する。
- donor exclusion、effective support、weighted MAD、shrink/clip/fallback parityを検証する。
- 正規Notebook採用とKaggle CPU実行は実装後も別承認とする。
- Stage 0全PASS時だけ1 variant / 773 HMM runsを再提示する。

## 進行中

- なし

## ブロック中

- 実装、Notebook採用、Kaggle実行、Stage 1、inference、submissionは未承認。
- 2026-07-24: exp362で同じK16 / k50 / bandwidth 500 ftの`n_eff>=10`が
  0/12,368 segmentsだったため、独立したtruth-free support非退化証拠が必要。

## 完了

- 2026-07-23: exp356として採番し、exp209 constant-rate親へ独立化した。
- 2026-07-23: exp323/355のrate meanを入力禁止にし、`sig_r,t`だけへ単一変更を固定した。
- 2026-07-23: Stage 0 calibration gate、Stage 1予約、SHA方針を確定した。
- 2026-07-24: exp362 support監査を依存証拠として追加し、exp356をblocked/demotedにした。
