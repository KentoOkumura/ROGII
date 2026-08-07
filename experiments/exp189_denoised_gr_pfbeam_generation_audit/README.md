# exp189_denoised_gr_pfbeam_generation_audit

## 目的

`denoised_gr_pfbeam_generation_audit` backlog を実装する。exp167 で FFT notch は弱かった一方、rolling median / Savitzky-Golay は GR matching surface の gap や entropy を改善したため、PF/Beam の observation likelihood に固定 smoothing を入れた場合に候補生成が改善するかを train-side で確認する。

## 状態

- `completed_train_side_diagnostic_no_submit`
- Kaggle train v1 完了。
- inference / submit は対象外。

## 仮説

raw GR の局所ノイズを smoothing すると、PF の likelihood weight と Beam の観測 cost が安定し、candidate RMSE、PF effective sample size、resampling 頻度、path jump、selector oracle headroom のいずれかが raw baseline より改善する。

## 範囲

- Route: `pf_beam`
- 親: `denoised_gr_pfbeam_generation_audit` backlog
- 参照: exp072 full replay feature cache、exp099 multi-observation audit、exp167 denoised GR audit、exp170 heel calibration audit
- 検証: exp072 の `TVT_input_missing_equivalent_exp063_rows`
- GR filter: raw、rolling median w11、Savitzky-Golay w31 p2
- 乱数: filter 間で同じ well / seed index の stable SHA256 seed を共有
- 対象外: FFT notch、heel calibration、LightGBM 学習、inference port、direct replacement、submit

## 検証方針

- exp072 train feature cache の `TVT_input_missing_equivalent_exp063_rows` を scoring surface として使う。
- target wells は true error を使わず、eval row / md_since coverage で最大 64 wells を選ぶ。
- raw / rolling median / Savitzky-Golay は同じ PF seed、particles、transition noise、beam width で比較する。
- `pf_raw_lik_mean` を primary baseline とし、filter delta、bucket/group/by-well、PF diagnostics、oracle headroom を見る。

## 所見

- PF likelihood smoothing は raw PF から大きく悪化したため不採用。
- Beam smoothing は raw Beam から小幅改善したが、既存 `exp072_pf_ancc` に届かず direct replacement には使わない。
- smoothed candidate の oracle headroom はあるため、使うなら selector / ML confidence feature 材料に限定する。

## 生成物

Kaggle train 実行後、`artifacts/` に candidate metrics、filter delta metrics、bucket/group/by-well metrics、PF diagnostics、target wells、row candidates、summary JSON を保存する。
