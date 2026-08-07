# exp217_grcal_public_raw_pf_confidence_features_on_exp158

## 状態

Route: `pf_beam`  
Status: `closed_train_side_positive_vs_exp158_not_anchor_no_submit`

Kaggle train v3 は `KernelWorkerStatus.COMPLETE`。best Viterbi は RMSE `10.669620824` で exp158 continuity `10.789163253` から `-0.119542` 改善したが、exp184 `10.560650325` / exp191 `10.598006880` には届かないため、inference / submit へ進めず、この実験は閉じる。

Kaggle train v1 は `KernelWorkerStatus.CANCEL_ACKNOWLEDGED` で停止。完走 metrics は生成されていないため、CV 評価や採用判断には使わない。

Kaggle train v2 も `KernelWorkerStatus.CANCEL_ACKNOWLEDGED` で停止。ログでは `GPU enabled: True` だったが、`[pubraw] 1/773` 以降で完走しておらず、CV 評価には使わない。

2026-07-13 に `pubraw_` 生成を cache stage として分離。`exp217_grcal_public_raw_pf_confidence_features_on_exp158_pfbeam_features.ipynb` が `id + pubraw_*` を `exp217_grcal_public_raw_pf_confidence_features_on_exp158_pubraw_features.csv.gz` に保存し、通常 train は `kentookumura/exp217-pubraw-cache-v1` を kernel source として読む構成にした。cache notebook は `KernelWorkerStatus.COMPLETE`。3,783,989 rows / 773 wells / 25 pubraw features を生成した。

## 仮説

`grcal_public_raw_pf_confidence_features_on_exp158` backlog の実装。exp158 の continuity selector と同じ 8 候補を維持し、exp214 の public-like raw PF diagnostics を full train の exp158 selector surface 上で再生成して `pubraw_` confidence feature として追加する。

`pubraw_pf_scale5/12` は候補値として選ばせない。direct replacement、blend、postprocess、PF weight replacement、inference、submit はこの実験の範囲外。

## 実装

- 親: `exp158_segment_continuity_selector_on_exp157`
- 候補集合: `pf_ancc`, `beam_mean`, `likpf_mean`, `sc_ens`, `hyb`, `tvt_dense`, `tvt_densew`, `tvt_dense50`
- 追加特徴: `pubraw_pf_scale5`, `pubraw_pf_scale12`, scale spread, GR residual sigma, ESS, resampling rate, seed weight concentration, candidate distance features
- PF config: 500 particles x 128 seeds、scales `[3, 5, 8, 12]`
- LightGBM: 3 configs x 5 folds = 15 boosters
- 後処理: exp158 と同じ Viterbi grid

## 検証方針

Kaggle train 後に、global RMSE、path switch、near `000_050`、`1000_plus`、worst-well regression、exp115 hidden-like subgroup、`pubraw_` feature importance / bucket readout を確認する。

## 所見

Kaggle train v1 は public raw PF feature generation 後、fold 0/1 のモデル保存まで進んだが、fold 2 学習開始直後に停止した。v2 は T4 GPU runtime でも停止。v3 は `kentookumura/exp217-pubraw-cache-v1` を kernel source として読み、pubraw 再生成を skip して完走した。`pubraw_` feature は feature importance 上位に入っており、exp158 への add-only confidence feature としては有効。ただし PF/Beam route の既存 reference には届かないため、採用候補にはしない。
