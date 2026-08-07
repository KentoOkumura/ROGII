# 要件

## 依頼

`denoised_gr_pfbeam_generation_audit` backlog を実装する。

## 制約

- Route: `pf_beam`
- exp167 で FFT notch は weak と判断済みのため、FFT notch は実行しない。
- exp170 で heel calibration は shift-scan top1 と PF/Beam observation rank を悪化させたため、calibration 依存では進めない。
- rolling median / Savitzky-Golay smoothing 単体の scoped train-side audit に限定する。
- exp072/099 と同じ `TVT_input_missing_equivalent_exp063_rows` を validation surface とする。
- target well / smoothing window / threshold は true error で選ばない。
- PF/Beam の seed、particles、beam width、transition config は raw と smoothed variants で固定する。
- LightGBM 学習、inference port、direct replacement、submit は対象外。
- 再現性: `docs/06_reproducibility.md` に従い、PF stochastic 処理と SHA 記録方針を設計に明記する。

## 受け入れ基準

- `experiments/exp189_denoised_gr_pfbeam_generation_audit/` に config、train/inference notebook、helper、記録ファイルがある。
- `config.yaml` の `experiment.route` は `pf_beam`。
- train notebook で raw / rolling median / Savitzky-Golay の filter contract、入力 cache、target wells、PF/Beam runtime が確認できる。
- helper は exp072 eval cache を読み、raw と smoothed GR likelihood を同じ seed / particles / beam width で比較する。
- candidate RMSE、filter delta、bucket/group/by-well metrics、PF ESS/resampling、path jump、oracle headroom を生成物として保存する。
- deterministic submission anchor として扱わず、submission SHA は不要。gzip row candidates は decompressed content SHA を主証拠として記録する。
