# 設計

## アプローチ

exp092 の U-projection correction / disagreement surface を維持し、その上に `tvt_dense` family の target-free confidence features を add-only で追加する。

exp135 では dense hard gate は global OOF を悪化させた。一方で PF `likpf_mean` worst50 と common PF+ML worst26 では `tvt_densew` が exp092 より良い場面があったため、dense candidate は「採用する予測」ではなく「exp092 が外れやすい regime の説明変数」として扱う。

## 実験範囲

- 対象実験: `exp151_tvt_dense_addonly_confidence_features_on_exp092`
- Route: `ml_model`
- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- 変更する変数: LightGBM feature surface に `tvt_dense_confidence` feature groups を追加する。
- 固定する変数: exp072 base feature cache、exp092 U-projection settings、target `TVT - last_known_tvt`、GroupKFold by well、LightGBM 3 config family。

## 特徴量

- `dense_confidence_geometry`: `md_since_norm`、`tail_rank_norm`、near flag、longtail flag。
- `dense_candidate_path`: `tvt_dense` / `tvt_densew` / `tvt_dense50` の drift、absolute drift、slope、roughness。
- `dense_candidate_disagreement`: dense family std/range、`likpf_mean` / `beam_mean` / `pf_ancc` と `tvt_densew` の差、dense pair absdiff、`pf_vs_dense`、`dense_std`、high-disagreement proxy。

## 再現性設計

- seed policy: GroupKFold seed 42。新規 PF/Beam 乱数は使わない。
- stochastic 処理の有無: upstream exp072 PF/Beam cache と GPU LightGBM training が stochastic component。feature merge 自体は deterministic。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。exp072 cache を読む。
- 並列処理と乱数の関係: LightGBM は `gpu_use_dp=true`、`deterministic=true`、`force_col_wise=true`、`n_jobs=8`、`num_threads=8`。
- train cache / test feature regeneration の SHA 記録方針: gzip は decompressed content SHA を主証拠にする。
- model manifest / prediction / submission SHA 記録方針: train manifest、OOF prediction SHA、model SHA を記録する。submission はこの実験では未作成。
- Kaggle package bootstrap 確認方針: push 前に `prepare-kaggle-notebooks --strict` を通し、生成 package の config と notebook metadata を確認する。

## リスク

- リークリスク: exp092 OOF prediction や true-error rank は特徴に入れない。dense scale は candidate drift/disagreement のみで作る。
- CV/LB 不一致リスク: dense confidence が train-specific regime に過適合する可能性がある。near-row、longtail、worst-well、exp115 hidden-like stress、raw-test parity を後続確認する。
- ランタイム/メモリリスク: exp092 surface + 30 add-only features、15 boosters なので exp149 相当以下のコスト。
- 再現性リスク: GPU LightGBM は bitwise anchor と扱わない。採用候補に進む場合のみ raw-test parity と saved booster inference を追加確認する。
