# 設計

## アプローチ

exp148 の feature surface を基準に、full-prefix 由来で TVT 急降下の影響を受けやすい learned multiobs 系だけを last50 crop-window 版へ差し替える。exp166 の全系統 replacement が悪化したため、今回は single-family replacement として損失を絞る。

last50 cache は専用 notebook で生成し、学習 notebook は cache を必須入力にする。LightGBM はタイムアウト対策として `lgb0` / `lgb1` / `lgb2` に分割する。

## 実験範囲

- 対象実験: `exp172_prefix_crop_last50_replacement_only_on_exp148`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 比較対象: exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960
- 変更する変数: prefix crop window (`last50`) と learned multiobs replacement-only feature list
- 固定する変数: exp072 replay cache、exp145 learned likelihood cache、GroupKFold split、LightGBM config family、CPU deterministic mode

## Replacement 設計

- 落とす learned multiobs 系列: `ll_multiobs_score_*`, `ll_multiobs_mae_*`, `ll_multiobs_ncc_*`
- 追加する crop group: `prefix_crop_last50_multiobs`
- 残す exp072/exp092 full-prefix 系列: `sc8_d`, `sc8_sc`, `sc15_d`, `sc15_sc`, `sc25_d`, `sc25_sc`, `sc_cons_d`, `sc_ens_d`, `sc_trust`, `cal_a`, `cal_b`, `pfx_rmse`, `slp_all`, `slp_z`, `slp_b_d_all`, `ktvt_range`, `ktvt_std`
- 残す learned likelihood 系列: learned probability / expected error / rank / entropy / candidate range など、multiobs 以外の confidence 列

## 再現性設計

- seed policy: fixed GroupKFold seed 42。新規 PF RNG は使わない。
- stochastic 処理の有無: なし。upstream PF/Beam cache と learned likelihood cache は既存 Kaggle artifact を参照する。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。exp072 / exp145 cache を読む。
- 並列処理と乱数の関係: feature build は deterministic。LightGBM は deterministic flags、`force_col_wise=true`、`num_threads=8`。
- CPU/GPU runtime: CPU のみ。
- train cache SHA: prefix crop cache manifest に raw gzip SHA と decompressed content SHA を記録する。
- model manifest / prediction SHA: split train output の manifest と prediction SHA を記録する。
- Kaggle package bootstrap: push 前に notebook metadata、kernel sources、`enable_gpu=false`、`enable_internet=false` を確認する。

## リスク

- リークリスク: crop は known prefix rows のみを使い、評価行 target を直接使わない。GroupKFold は well grouped。
- CV/LB 不一致リスク: exp148 が Public LB anchor のため、CV 改善だけで submit しない。raw-test/current-test parity と worst-well を確認する。
- ランタイム/メモリリスク: feature cache 生成は重いが、学習側は variant に必要な last50 列だけを `usecols` で読み込む。学習は `lgb0` / `lgb1` / `lgb2` に分割する。
- 再現性リスク: Kaggle CLI logs は完了まで空の前提。logs 空だけで再 push しない。
