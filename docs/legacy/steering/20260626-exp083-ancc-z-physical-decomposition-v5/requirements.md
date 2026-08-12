# 要件

## 依頼

`exp083_pf_beam_true_tvt_2d_well_eda` に v5 の追加診断 plot を作る。目的は PF 生成結果の急変原因ではなく、train true TVT の急変が well trajectory の `Z` 変化、raw train-only formation top `ANCC` の傾き、またはその合成 `ANCC - Z` に対応するかを目視できるようにすること。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- raw `ANCC` / `ASTNU` / `ASTNL` / `EGFDU` / `EGFDL` / `BUDA` は train horizontal CSV にだけあるため、train EDA 専用にする。推論特徴、提出候補、hard router には直接使わない。
- 既存 exp072 feature cache と raw train CSV は `id={well}_{row_idx}` / well id で align し、行順の暗黙 join だけに依存しない。
- 既存 v4 clean all-well plot 生成物は履歴として残し、v5 は別 prefix で保存する。

## 受け入れ基準

- 各 well plot に `true TVT` と `pf_z` が表示される。
- 各 well plot に `Z`、`ANCC`、可能な場合は `ASTNU` / `ASTNL` / `EGFDU` / `EGFDL` / `BUDA` が表示される。
- 各 well plot に `dTVT/dMD`、`-dZ/dMD`、`dANCC/dMD`、`d(ANCC - Z)/dMD` が表示される。
- `abs(dTVT/dMD)` が大きい箇所を縦線で示し、急変箇所と `Z` / `ANCC` / `ANCC-Z` の傾きの対応を読める。
- deterministic anchor として扱わず、model SHA、prediction SHA、submission SHA は対象外として記録する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
