# exp103_pf_z_xy_likpf_ensemble_parity 結果

## 状態

Kaggle train v4 完了。

## 目的

exp100 `pf_z_xy_slope` を単発 PF 候補ではなく、exp072 `lik_pf` と同じ 128 seed likelihood-weighted ensemble として比較する。

## 実行条件

- Kernel: `kentookumura/exp103-pf-z-xy-likpf-ensemble-train`
- Version: 4
- Status: `KernelWorkerStatus.COMPLETE`
- runtime: 22252.17 sec
- rows: 3,783,989
- wells: 773
- n seeds: 128
- particles: 500
- n jobs: 4

## Candidate Metrics

| candidate | RMSE | MAE | within10 | bias |
| --- | ---: | ---: | ---: | ---: |
| `exp072_likpf_mean` | 11.594898 | 7.067633 | 0.772807 | -1.099423 |
| `xy_likpf_scale_12` | 13.916271 | 8.554011 | 0.705260 | -0.970944 |
| `xy_likpf_scale_8` | 13.961015 | 8.577330 | 0.701849 | -0.926790 |
| `xy_likpf_scale_5` | 14.030092 | 8.615191 | 0.700313 | -0.896577 |
| `xy_likpf_scale_3` | 14.092584 | 8.650626 | 0.698493 | -0.862208 |
| `xy_likpf_mean` | 14.580554 | 9.664990 | 0.650353 | -1.050961 |
| `exp072_pf_z` | 17.788171 | 10.677487 | 0.647668 | -0.934560 |

## 判定

best XY は `xy_likpf_scale_12`。exp072 `pf_z` には RMSE -3.871901、within10 +0.057592 で勝った。一方、exp072 `likpf_mean` には RMSE +2.321373、within10 -0.067548 で明確に負けた。

したがって、`xy_likpf_*` は direct inference port / submit 候補にしない。exp100 の XY slope idea は seed ensemble 化と likelihood-weighted scale 化をしても `likpf_mean` を置き換えられない。

ただし selector / ML 特徴量候補としては残す。`likpf_mean + exp072_pf_z + xy_likpf_scale_12` の oracle は RMSE 7.808425 / within10 0.896735 で、`likpf_mean + exp072_pf_z` oracle RMSE 9.115201 / within10 0.861225 から headroom が増える。`xy_likpf_scale_12`、`xy_likpf_seed_std`、`xy_likpf_scale_12 - exp072_likpf_mean`、`xy_likpf_scale_12 - exp072_pf_z`、path smoothness は guarded selector または ML add-only feature の候補として扱う。

## 生成物

- `artifacts/exp103_pf_z_xy_likpf_ensemble_parity_candidate_metrics.csv`
- `artifacts/exp103_pf_z_xy_likpf_ensemble_parity_bucket_metrics.csv`
- `artifacts/exp103_pf_z_xy_likpf_ensemble_parity_by_well.csv`
- `artifacts/exp103_pf_z_xy_likpf_ensemble_parity_xy_likpf_quality.csv`
- `artifacts/exp103_pf_z_xy_likpf_ensemble_parity_candidate_wide.csv.gz`
- `artifacts/exp103_pf_z_xy_likpf_ensemble_parity_candidate_long.csv.gz`
- `artifacts/exp103_pf_z_xy_likpf_ensemble_parity_summary.json`

## SHA

- `candidate_metrics.csv`: `5570a058351aa460425bc3504159b225819ea163a6fd5a52669107470a140b9f`
- `candidate_wide.csv.gz`: `bd8ef95b6cdaf029c13bd843e50b552dc68921ed8a24d2775bb1f54a3bda8466`
- `candidate_wide.csv.gz` decompressed: `68974d20eaca854c0458ecb234c9c569daf4d66764f1a788083b74cc9359a012`
- `candidate_long.csv.gz`: `a7b9bca2c67ce8d3c122f642b4f2a99855a57f631eb86d38a125b4f71f949ad7`
- `candidate_long.csv.gz` decompressed: `b087071b619c3de820058920e2b985b22e7580205561677957c89485b03625a1`

## 次アクション

PF 側を続けるなら、`xy_likpf_scale_12` は direct candidate 置換ではなく selector 候補 / ML add-only feature として使う。候補生成そのものを続ける場合は、strict exp072 `pf_z` parity の multi-seed cache や observation likelihood 改善に戻す。
