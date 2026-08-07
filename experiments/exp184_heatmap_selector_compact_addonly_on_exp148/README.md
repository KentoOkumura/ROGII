# exp184_heatmap_selector_compact_addonly_on_exp148

## 目的

exp148 の ML route anchor に、exp184 の CNN/SDF/MTP heatmap selector signal を compact add-only feature として追加する。

## 仮説

exp184 の heatmap selector は PF/Beam route では train-side positive だった。一方で exp148 の ML anchor は強く、exp188/exp194 では広い selector confidence block や replacement が悪化した。したがって、既存 `learned_likelihood_confidence` を維持したまま、heatmap/selector の信頼度を少数列だけ追加すれば、ML model が局所的な不確実性を補助 signal として使える可能性がある。

## 親と比較対象

- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- heatmap / selector 親: `exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158`
- heatmap source: `exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe`
- 比較: exp148 historical CV/Public LB、exp188 add-only negative、exp194 replacement negative、exp190/195 DCM 系

## 方針

`projection_correction`、`u_disagreement`、`learned_likelihood_confidence` は維持し、`heatmap_selector_compact` だけを追加する。exp184 selected TVT は direct replacement、blend、postprocess、hard gate、submit candidate として使わない。

追加 feature は selected candidate/family、selected vs `likpf_mean`、selected vs exp148 OOF、segment stability、exp182 heatmap score/margin/entropy、sparse sample distance、real-vs-control confidence gap に絞る。exp184 の heatmap path features 全量投入はしない。

## 検証方針

GroupKFold 5 folds、LightGBM 3 configs で exp148 historical CV と比較する。Kaggle train は CPU (`cpu_deterministic_threads8`, `enable_gpu=false`) とし、timeout 対策として `train_lgb0` / `train_lgb1` / `train_lgb2` に分割する。各 notebook は 1 config x 5 folds = 5 boosters だけを学習する。

control / parent は再学習しない。global OOF、worst-well、tail rank、distance bucket、heatmap sparse distance bucket、feature importance を確認する。split 3本の output を取得した場合だけ、横断 `lgb_mean` ensemble CV を後処理で計算する。

## 所見

Kaggle CPU split train version 1 は 3本とも完了したが、cross-split `lgb_mean` は RMSE 8.604130846 で、exp148 GPU historical `lgb_mean` 8.501281182 から +0.102849664、exp148 CPU runtime `lgb_mean` 8.528698114 から +0.075432732 悪化した。exp188 add-only 8.539573790 も下回るため、train-side negative として採用しない。

split 別:

- `lgb0`: 8.710685277
- `lgb1`: 8.639432353
- `lgb2`: 8.611075285
- cross-split `lgb_mean`: 8.604130846

Kaggle run では split train inputs に exp148 train output がなかったため、optional exp148 OOF delta features は unavailable となり、local smoke の 31 hmp184 features ではなく 28 hmp184 features で完了した。厳密な 31-feature rerun は可能だが、今回の CV 悪化幅が大きいため現時点では実行しない。

## 状態

`completed_train_side_rejected_no_submit`。metadata は 3本とも `enable_gpu=false`。実行対象は active variant 1、CPU split train 3本、各 5 boosters、合計 15 boosters、parent/control 再学習なし。inference port と submit は行わない。
