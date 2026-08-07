# exp114_spatial_neighbor_prior_signal_audit

## 状態

- ルート: ensemble
- 状態: completed_train_side_audit_supported_no_submit
- CV: 11.151818387
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-23
- 親実験: `exp099_pf_multi_observation_likelihood_probe`

## 仮説

X/Y が近いだけでなく掘削方向、軌跡形状、local tangent、dZ/dMD、prefix TVT range が似た train-fold wells から作る spatial neighbor TVT drift prior は、PF/Beam/likPF の誤差方向を説明する診断信号になり得る。

## 変更点

- exp099 train feature cache を固定入力として読む。
- raw train horizontal well から centroid、start/end、bbox、azimuth、local azimuth、tortuosity、dZ/dMD、prefix TVT range を well-level geometry として作る。
- `xy_only_k8`、`xy_plus_azimuth_k8`、`xy_plus_trajectory_shape_k8`、`xy_plus_direction_and_typewell_k8` を fold-safe に比較する。
- prior-only、`likpf_mean` / `pf_ancc` / `beam_mean` への clipped correction、base error と prior-base 差分の相関・符号一致率を保存する。

## 検証方針

- Fold: 5-fold fixed seed GroupKFold 相当の well split
- Group: `well`
- Stratification: なし
- Leakage Check: validation well と同 fold validation wells の true TVT は neighbor source に入れない。query には visible prefix と target-free geometry / native typewell group だけを使う。

## 実行入口

- 学習 notebook: `exp114_spatial_neighbor_prior_signal_audit_train.ipynb`
- 推論 notebook: `exp114_spatial_neighbor_prior_signal_audit_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp114_spatial_neighbor_prior_signal_audit`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV / best RMSE | 11.151818387 |
| best candidate | `xy_plus_trajectory_shape_k8_likpf_mean_corr_a0p2_c40` |
| delta vs `likpf_mean` RMSE | -0.443079285 |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- `xy_plus_trajectory_shape_k8` prior correction は `likpf_mean` から RMSE -0.443079 改善。
- 全 distance bucket で RMSE 改善。
- OOF prior coverage は `xy_plus_trajectory_shape_k8` で 0.998525。

### 悪かった点

- by-well は 416 改善 / 357 悪化で、最大悪化 +6.508121 RMSE が残る。
- direct correction / submit には使わない。

### リスク / 注意

- X/Y 近傍効果と native typewell group 効果を混同しやすい。
- global RMSE が良くても worst-well regression が残る可能性がある。
- この実験では inference port / submit は選ばない。
- Kaggle train output の OOF gzip は 880MB あり、必要時だけ参照する。

## 次

- `spatial_neighbor_prior_confidence_gate_on_exp092` として、spatial prior を信用してよい row/well を判定する confidence / gate follow-up を検討する。
- ML に特徴量として入れる評価は `spatial_neighbor_prior_ml_features_on_exp092` として別 backlog に分ける。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
