# 要件

## 依頼

`KAGGLE_DIRECTION.md` のバックログ `heel_calibrated_shift_scan_pfbeam_audit` を `exp170_heel_calibrated_shift_scan_pfbeam_audit` として実装する。

Georgy Mamarin の公開 notebook §5c で示唆されている「known heel の `TVT_input` で GR gain/offset を calibration すると datum localization が戻る」という仮説を、ML特徴量化やPF/Beam生成変更の前に train-side で検証する。

## 制約

- Route: `pf_beam`
- calibration は known prefix の `TVT_input` と horizontal GR だけで fit する。
- hidden/eval tail の true `TVT`、oracle best candidate、candidate true error、true-error rank を calibration、threshold、mode selection に使わない。
- PF/Beam / likelihood-PF は再実行しない。exp072 の固定 train pseudo-tail cache は candidate observation likelihood readout にだけ使う。
- direct candidate replacement、hard selector、exp148 ML add-only feature 化、inference port、submit は含めない。
- 再現性: `docs/06_reproducibility.md` に従い、upstream exp072 cache、入力 raw file SHA、gzip decompressed SHA、Kaggle kernel version の扱いを記録する。

## 受け入れ基準

- `experiments/exp170_heel_calibrated_shift_scan_pfbeam_audit/` が作成されている。
- `config.yaml` に `experiment.route: pf_beam`、lineage、leakage policy、再現性方針が記録されている。
- train notebook が、raw / flat-calibrated / heel-calibrated の shift-scan metrics を生成する。
- fixed PF/Beam candidates (`pf_ancc`, `beam_mean`, `likpf_mean` など) について、candidate RMSE と observation cost / rank / gap readout を出す。
- metrics は all、hidden_tail、prefix_backtest、near-row、`1000_plus`、well別で読める。
- `make validate-exp EXP=exp170_heel_calibrated_shift_scan_pfbeam_audit` が通る。
- deterministic anchor としては扱わず、submission / model SHA は対象外であることを明記する。
