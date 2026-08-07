# exp179_cnn_sdf_mtp_heatmap_probe

## 状態

- ルート: ml_model
- 状態: completed_train_side_gpu_probe_supported
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-03
- 親実験: `mtp_heatmap_sdf_mdn_probe` backlog / discussion 699853

## 仮説

discussion 699853 の 5ch heatmap CNN/SDF/MTP は、GR waveform と観測済み TVT_input prefix から、target-free な typewell TVT window 内に真値近傍の trajectory mode を残せる可能性がある。最初の合格条件は単一 TVT RMSE や提出ではなく、well GroupKFold の pseudo-tail で real GR が shuffled-GR / no-GR control より topK coverage を改善すること。

## 変更点

- 5ch image: `t_gr`、`h_gr`、`t_gr-h_gr`、observed `TVT_input` prefix 由来 SDF history、prefix mask。
- typewell window center は true TVT ではなく `last_known_tvt - (Z - last_known_z)` の flat prior で作る。
- 小型 PyTorch CNN が `K=10` の path head を出し、closest-mode CE で学習する。
- `real_gr`、`shuffled_gr`、`no_gr` の 3 variants を同じ sample schedule / fold / epoch で比較する。
- inference port と submission は作らない。

## 検証方針

- Fold: GroupKFold 5 split の fold 0
- Group: well id
- Stratification: target-in-grid、distance from prefix、well 別 coverage、GR control variant
- Leakage Check: valid true TVT は label / metric のみに使う。input heatmap、normalization、typewell window center、sample schedule、negative controls には使わない。

## 実行入口

- 学習 notebook: `exp179_cnn_sdf_mtp_heatmap_probe_train.ipynb`
- 推論 notebook: `exp179_cnn_sdf_mtp_heatmap_probe_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp179_cnn_sdf_mtp_heatmap_probe`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |

Kaggle train v2 / T4 GPU で完了。valid 512 samples / 32 wells、train 2,304 samples / 96 wells。

| variant | top1 within10 | top3 within10 | top5 within10 | top10 within10 | top10 oracle RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `real_gr` | 0.210938 | 0.449219 | 0.636719 | 0.794922 | 14.071006 |
| `shuffled_gr` | 0.101562 | 0.232422 | 0.328125 | 0.541016 | 21.395476 |
| `no_gr` | 0.062500 | 0.062500 | 0.062500 | 0.062500 | 136.025391 |

`real_gr` は top3 within10 で shuffled-GR を +0.216797、no-GR を +0.386719 上回った。

## 所見

### 良かった点

- real GR が shuffled-GR / no-GR controls を明確に上回った。
- target-free flat prior の grid は valid target を全サンプルで含んだ。
- T4 GPU で 3 variants x 5 epochs が約 2 分で完了した。

### 悪かった点

- v1 は Kaggle が P100 を割り当て、PyTorch 2.10 の CUDA build が `sm_60` 非対応だったため失敗した。v2 は T4 明示で解消した。
- まだ 1 fold / small window / small wells の smoke であり、full-fold や full-length inference の証拠ではない。

### リスク / 注意

- GPU training は bitwise deterministic anchor として扱わない。
- window center は target-free prior のため、true TVT が grid 外に出る行では coverage が上がらない。これは候補生成 failure として別途記録する。
- real GR が shuffled/no-GR と差がない場合、この MTP/CNN/SDF 方向は GR を使えていないと判断して閉じる。

## 次

- full-fold / larger-window / geometry-channel ablation を次候補にする。
- direct TVT replacement、inference port、submission はまだ行わない。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
