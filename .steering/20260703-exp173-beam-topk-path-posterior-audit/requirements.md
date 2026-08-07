# 要件

## 依頼

`KAGGLE_DIRECTION.md` のバックログ `beam_topk_path_posterior_audit` を実装する。

## 制約

- Route: `pf_beam`
- train-side pseudo-tail audit に限定し、inference port / submit はしない。
- exp072 cache には Beam の retained top-K path/cost がないため、Beam search を再実行して保存する。
- Beam generation、posterior temperature、diagnostic feature generation に evaluation-zone true TVT を使わない。
- true TVT は candidate RMSE、top-K oracle headroom、bucket readout の scoring にだけ使う。
- 再現性: `docs/06_reproducibility.md` に従い、PF/Beam 系の seed / SHA / gzip decompressed content SHA の扱いを設計に明記する。

## 受け入れ基準

- `experiments/exp173_beam_topk_path_posterior_audit/` に実験 config、実行 module、train notebook、記録ファイルがある。
- Beam の retained top-K path と final path cost から、top1/top2 commit、top-K weighted mean、固定温度 posterior mean、cost gap、entropy、path separation、top-K oracle headroom を出力できる。
- 比較対象として exp072 cache の `pf_ancc`、`beam_mean`、`likpf_mean` を materialize できる。
- 出力予定は candidate metrics、bucket metrics、by-well metrics、group metrics、beam quality、top-K diagnostics、candidate wide、summary JSON に分かれている。
- Kaggle push 前の計算規模として、Beam variants 3 個、LightGBM config 0 個、fold 0 個、booster 0 本、control 再学習なしが記録されている。
- deterministic anchor として扱わない理由が `SESSION_NOTES.md` と `config.yaml` に書かれている。
