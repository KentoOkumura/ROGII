# 要件

## 依頼

`KAGGLE_DIRECTION.md` のバックログ `fft_denoised_gr_matching_audit` を
`exp167_fft_denoised_gr_matching_audit` として実装する。

GR sensor rotation 由来の周期ノイズを FFT notch / band-stop 系の target-free denoise で落とすと、
typewell GR matching / shift scan の localization surface が raw GR より改善するかを先に監査する。
これは ML 特徴量追加や PF/Beam 生成変更ではなく、matching quality の train-side diagnostic として扱う。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- raw train の horizontal GR、typewell GR、MD、known `TVT_input` prefix のみを matching 入力に使う。
- eval tail の true TVT は評価指標にだけ使い、denoise 周波数選択、normalization、shift center、threshold には使わない。
- PF/Beam 候補生成、exp148 ML add-only feature、direct replacement、submission はこの実験では行わない。

## 受け入れ基準

- `config.yaml` に route、lineage、audit 対象 filter、shift scan 範囲、評価 scope、再現性方針が明記されている。
- train notebook から raw / rolling median / Savitzky-Golay fallback / FFT notch の matching scan を実行できる。
- 生成物として row context、filter metrics、bucket metrics、well metrics、raw-vs-denoised gain、input summary、summary JSON を保存する。
- 評価は prefix backtest、hidden tail sampled audit、top1 within2/5/10ft、top1-top2 gap、entropy、±15-25ft decoy gap、raw vs denoised localization gain、near-row、`1000_plus`、worst-well を含む。
- stochastic 処理なしであること、PF/Beam 生成や GPU 学習がないこと、Kaggle bootstrap package を prepare 後に使うことが `SESSION_NOTES.md` に記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録できる。
