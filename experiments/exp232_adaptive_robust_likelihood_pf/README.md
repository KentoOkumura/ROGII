# exp232_adaptive_robust_likelihood_pf

## 状態

- Route: `pf_beam`
- 状態: `completed_train_side_rejected_no_inference_no_submit`
- 親: `exp072_exp063_full_replay_feature_cache`
- control: exp209 enriched cache から復元する exp072 `likpf_mean`（`T=1`、再生成しない）

## 仮説

局所 GR motif が particle prediction と強く矛盾し、かつ change-point / novelty / particle collapse の裏付けがある row だけ観測尤度を緩めれば、誤った mode への collapse を減らし、long-tail の粒子区間 coverage を改善できる。

## 検証方針

- 全 eligible well の exp072-compatible pseudo-tail を評価する。
- 新規 variant は `temp_t2` と `temp_t4` のみ。gate 外では `T=1` の既存 Gaussian 尤度を厳密に使う。
- particle数 500、128 seeds、raw GR/typewell GR、遷移、resampling、seed mean aggregation は固定する。
- overall、distance bucket、exp115 hidden-like、worst well、ESS、resampling、gate率、固定サンプルの weighted particle p05-p95 coverage、first sampled loss を保存する。
- outlier mixture、inference、submission はこの実験に含めない。

## 所見

初回は exp072 ML feature cache に `likpf_mean` が含まれず、PF 実行前に停止した。ユーザー指定により exp209 enriched cache の `hmm_mean_tvt - hmm_minus_likpf_mean` で `likpf_mean` を復元し、ID、well、target、last known TVT、`md_since` の一対一整合を検査して比較 control にだけ使った。canonical v3 は timeout/cancel 後に分割し、T=2/T=4 はそれぞれ独立 CPU kernel で完走した。

| candidate | RMSE | control差 | 1000_plus | 最大 well 悪化 | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| exp072 `likpf_mean` | 11.594898 | 0.000000 | 12.704015 | 0.000000 | control |
| `temp_t2` | 13.529887 | +1.934989 | 14.775089 | +45.905685 | 不採用 |
| `temp_t4` | 13.532730 | +1.937833 | 14.778864 | +45.706171 | 不採用 |

gate は非常に稀でも、両 variant とも long-tail と worst well を大きく悪化させた。inference と submission は生成しない。

## 次のアクション

この temperature-only direct PF update は終了する。robust likelihood を再検討するなら、温度 grid の再実行ではなく、gate 発火後の長期 path divergence を診断して containment guard を先に作る。

## 参照ファイル

- `config.yaml`
- `adaptive_robust_likelihood_pf.py`
- `exp232_adaptive_robust_likelihood_pf_train.py`
- `exp232_adaptive_robust_likelihood_pf_train_variant0.py`
- `exp232_adaptive_robust_likelihood_pf_train_variant1.py`
- `SESSION_NOTES.md`
