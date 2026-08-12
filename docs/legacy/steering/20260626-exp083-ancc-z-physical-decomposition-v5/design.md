# 設計

## アプローチ

既存 exp083 の exp072 feature cache 読み込みをそのまま使い、追加で `data/raw/train/{well}__horizontal_well.csv` から raw physical columns を読み込む。feature cache の `id` 末尾 row index と `well` を使って raw train row に join し、`Z`、formation top、`true TVT`、`pf_z` を同じ `MD` 軸で 3 段 plot にする。

plot は次の構成にする。

- 上段: `true TVT`、`pf_z`、参照用 `last_anchor_tvt`
- 中段: raw `Z`、raw `ANCC`、raw `ASTNU` / `ASTNL` / `EGFDU` / `EGFDL` / `BUDA` の薄線
- 下段: `dTVT/dMD`、`-dZ/dMD`、`dANCC/dMD`、`d(ANCC - Z)/dMD`

`abs(dTVT/dMD)` の上位 quantile を急変点として縦線表示する。

## 実験範囲

- 対象実験: `exp083_pf_beam_true_tvt_2d_well_eda`
- Route: `pf_beam`
- 親実験: `exp072_exp063_full_replay_feature_cache`
- 変更する変数: 可視化のみ。raw train physical context join、physical decomposition plot、v5 output prefix。
- 固定する変数: exp072 source cache、既存 candidate materialization、representative/all-well selection、Kaggle CPU runtime、提出なし。

## 再現性設計

- seed policy: 新規乱数なし。既存 all-well selection では seed 不使用。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: 既存 exp072 cache を読むだけで新規生成しない。
- 並列処理と乱数の関係: raw CSV join と plot のみで RNG なし。
- CPU/GPU runtime と deterministic flags: Kaggle CPU notebook。GPU 不使用。
- train cache / test feature regeneration の SHA 記録方針: exp072 gzip input は既存と同じ raw SHA と decompressed SHA を summary に記録する。
- model manifest / prediction / submission SHA 記録方針: model、prediction、submission は生成しないため対象外。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks --strict` 後、train package に更新済み `config.yaml` / `pf_beam_true_tvt_eda.py` / notebook が含まれることを確認する。

## リスク

- リークリスク: raw formation columns と true TVT を train EDA に使う。推論用特徴や提出判断に直接使うと漏れになるため、生成物の解釈を診断用途に限定する。
- CV/LB 不一致リスク: EDA であり CV/LB 改善を主張しない。
- ランタイム/メモリリスク: 773 well の raw CSV を読む。必要列だけ読み、cache 側の対象 row index に絞って join する。
- 再現性リスク: plot PNG の描画順や matplotlib minor version で見た目は変わり得る。入力 SHA と manifest を主証拠にする。
