# exp132_multi_scale_gr_observation_likelihood 結果

## 状態

Kaggle train v1 完了。multi-scale GR observation likelihood は direct scorer / low-switch gate / immediate feature follow-up としては棄却。

## 仮説

exp099 の raw multi-observation score は oracle headroom を増やしたが、direct top1 scorer としては崩壊した。multi-scale な GR 観測尤度を target-free confidence / verifier feature として再設計すれば、直接候補置換ではなく低頻度 gate や exp092 系 ML feature の材料として使える可能性がある。

## 実装内容

- exp072 fixed candidate cache を読む train-side diagnostic。
- `pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb` を固定候補として使用。
- GR scorer は `[-48,-24,-12,0,12,24,48]` offset、window `[5,11,21]`、smoothed GR、local z-score、derivative、energy、NCC、decoy gap を使う。
- 診断候補として `msgr_top1`、`msgr_top2`、softmax、`likpf_msgr_blend`、low-switch gate を保存。
- exp072-style wide train feature cache を出力。

## 実行結果

- Kernel: `kentookumura/exp132-msgr-likelihood-train` v1
- Output: `experiments/exp132_multi_scale_gr_observation_likelihood/kaggle/output/train_v1`
- Rows / wells: 3,783,989 / 773
- Runtime: 2328.92 sec
- Best candidate: `likpf_mean` RMSE 11.594897 / MAE 7.067633 / within10 0.772807
- Best low-switch gate: `msgr_gate_m0p08_s0p45_d40` RMSE 11.632677 / within10 0.771381
- Gate delta vs `likpf_mean`: RMSE +0.037780 で悪化
- Direct scorer: `msgr_top1` RMSE 86.806694 で崩壊
- Softmax / blend: `likpf_msgr_blend_w0p1` RMSE 14.404808、`likpf_msgr_blend_w0p25` RMSE 24.392874 で不採用
- baseline+msgr oracle: RMSE 6.949725 / within10 0.921029。baseline primary oracle RMSE 7.434030 / within10 0.906525 からは headroom がある
- candidate rank score top1: `beam_mean` を選び RMSE 86.806694 で失敗
- best gate の by-well は 226 改善 / 528 悪化 / 19 同値
- best gate は全 distance bucket で `likpf_mean` より悪化

## 生成物

- feature cache: `artifacts/exp132_multi_scale_gr_observation_likelihood_multi_scale_gr_observation_likelihood_train_features.csv.gz`
- feature rows / wells / count: 3,783,989 / 773 / 75
- train feature gzip SHA256: `3ced89e1837321ea15fd22848aacde7ec8729aa7d97aae142c32fe5ff21124eb`
- train feature decompressed SHA256: `76f41392f0148d14568b87bd973d74da7c48a879ff20b1b2273a13db96606756`
- feature schema SHA256: `e54c55717916aee9c71f63917f891042cd4a7b6c11df1d6cd887c3602753686f`

## 判定基準

direct top1 の RMSE 改善は主判定にしない。以下を満たす場合のみ follow-up に進める。

- baseline+msgr の oracle/topK coverage が baseline primary より改善する。
- `msgr_gate_*` の best が `likpf_mean` より悪化しない、または悪化が小さく特定 bucket の confidence 診断として意味がある。
- near-row、1000+ longtail、worst-well regression を壊さない。
- feature cache が exp092 系 add-only feature として raw-test-compatible に使える。

## 判定

棄却。oracle top10 には headroom が残るが、非 oracle の rank score、direct top1、blend、low-switch gate が `likpf_mean` を上回れなかった。特に best gate が全 distance bucket と 528 wells で悪化しており、即時の inference port、submit、exp092 add-only feature 化には進めない。

feature cache は保存するが、用途は今後の learned verifier が hand-crafted negative / decoy signal を必要とする場合の診断材料に限定する。
