# exp132_multi_scale_gr_observation_likelihood

## 状態

- ルート: pf_beam
- 状態: completed_train_side_rejected
- CV: train-side pseudo-tail audit RMSE 11.594897 best remains `likpf_mean`
- Public LB: なし
- Private LB: なし
- Submit ID: なし
- 作成日: 2026-06-26
- 親実験: `exp099_pf_multi_observation_likelihood_probe`

## 仮説

exp099 の multi-observation likelihood は oracle headroom を増やしたが、direct top1 / softmax blend は崩壊した。単点 GR 差に近い scorer ではなく、複数 window、複数 offset、smoothed GR、local z-score、derivative、energy、decoy gap を使った multi-scale GR observation likelihood にすれば、候補を直接置換せずに verifier / confidence feature として使える可能性がある。

## 変更点

- exp072 deterministic full replay train cache を固定入力として読む。
- `pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb` を multi-scale GR observation likelihood で再採点する。
- 診断候補として `msgr_top1`、`msgr_top2`、softmax weighted candidates、`likpf_msgr_blend`、低頻度 switch gate を作る。
- exp072 と同じ wide feature cache 形式で、msgr score / MAE / NCC / derivative / energy / decoy gap などを保存する。
- true TVT は candidate RMSE / oracle / rank metrics の scoring にだけ使う。
- PF/Beam 再実行、supervised ranker、inference port、提出は行わない。

## 検証方針

- Fold: train-side pseudo-tail audit の既存 cache に従う
- Group: well
- Stratification: distance / tail rank / eval length / PF seed std / likPF delta / msgr score / msgr gap
- Leakage Check: multi-scale GR observation likelihood は raw horizontal GR、row index、finite prefix TVT_input、既存候補 TVT のみで計算する
- 合格条件: direct top1 ではなく、topK coverage、low-switch gate、exp092 系 add-only confidence feature 化の余地で判断する

## 実行入口

- 学習 notebook: `exp132_multi_scale_gr_observation_likelihood_train.ipynb`
- 推論 notebook: `exp132_multi_scale_gr_observation_likelihood_inference.ipynb`
- Kaggle 準備: `make prepare-kaggle-notebooks EXP=exp132_multi_scale_gr_observation_likelihood EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp132-msgr-likelihood-train --title 'exp132 msgr likelihood train' --run-on-push --strict"`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 期待生成物

- `exp132_multi_scale_gr_observation_likelihood_multi_scale_gr_observation_likelihood_train_features.csv.gz`
- `exp132_multi_scale_gr_observation_likelihood_multi_scale_gr_observation_likelihood_feature_schema.csv`
- `exp132_multi_scale_gr_observation_likelihood_candidate_metrics.csv`
- `exp132_multi_scale_gr_observation_likelihood_rank_metrics.csv`
- `exp132_multi_scale_gr_observation_likelihood_bucket_metrics.csv`
- `exp132_multi_scale_gr_observation_likelihood_by_well.csv`
- `exp132_multi_scale_gr_observation_likelihood_summary.json`

## 所見

- Kaggle train v1 は 3,783,989 rows / 773 wells で完了した。
- Best candidate は既存 `likpf_mean` RMSE 11.594897 / within10 0.772807 のまま。
- Best low-switch gate `msgr_gate_m0p08_s0p45_d40` は RMSE 11.632677 で `likpf_mean` から +0.037780 悪化した。
- `msgr_top1` は RMSE 86.806694、softmax / blend も大きく悪化し、direct scorer として崩壊した。
- baseline+msgr oracle は RMSE 6.949725 / within10 0.921029 で headroom はあるが、非 oracle の verifier が選べていない。
- best gate は 226 wells 改善 / 528 wells 悪化 / 19 同値で、全 distance bucket でも悪化した。
- feature cache は保存するが、即時の inference port、submit、exp092 add-only feature 化には進めない。

## 次

1. `multi_scale_gr_observation_likelihood` backlog は完了扱いで閉じる。
2. GR window 類似度系の追加実験は、exp132 の負結果を前提に優先度を下げる。
3. 保存済み feature cache は、将来の learned verifier が hand-crafted decoy / ambiguity signal を必要とする場合だけ参照する。
