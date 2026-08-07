# exp106_strict_exp072_pf_z_multiseed_scale_cache 結果

## 状態

Kaggle train v3 完了。提出なし。

## 目的

exp072 の元 `pf_z` を同一ロジックのまま seed 1 parity 再生成し、parity 通過後に multi-seed / scale cache として評価する。

## 判定

部分採用。strict parity は完全一致で通り、best multiseed 候補 `pf_z_ms_scale_3` は exp072 plain `pf_z` の代替候補、および selector の候補 path として有効。ただし `exp072_likpf_mean` には大きく届かないため、PF/Beam の単体主力候補や submit 候補としては採用しない。

| candidate | RMSE | MAE | within10 | bias |
| --- | ---: | ---: | ---: | ---: |
| `exp072_likpf_mean` | 11.594898 | 7.067633 | 0.772807 | -1.099423 |
| `pf_z_ms_scale_3` | 16.145943 | 9.155580 | 0.708807 | -0.752507 |
| `exp072_pf_z` | 17.788171 | 10.677487 | 0.647668 | -0.934560 |
| `strict_pf_z_parity_seed` | 17.788171 | 10.677487 | 0.647668 | -0.934560 |

parity summary: rows 3,783,989 / wells 773 / max_abs_diff 0.0 / rmse_diff 0.0。`pf_z_ms_scale_3` は `exp072_pf_z` から RMSE -1.642228、within10 +0.061139 改善しており、`pf_z` column の replacement / add-only feature として次段評価する価値がある。また、全体平均では `likpf_mean` より弱くても、一部 row / bucket / well で選択価値がある可能性があるため、selector candidate path として oracle selection rate、`likpf_mean` との disagreement、bucket 別改善、path continuity を確認する価値がある。一方で `exp072_likpf_mean` より RMSE +4.551045 悪いため、`likpf_mean` の単体代替、direct inference port、提出候補にはしない。

selector 候補としては exp103 より弱い。`likpf_mean + exp072_pf_z + pf_z_ms_scale_3` の oracle は RMSE 8.123717 / within10 0.887105 で、exp103 の `xy_likpf_scale_12` 追加 oracle RMSE 7.808425 / within10 0.896735 には届かない。exp106 は strict parity と安定した `pf_z` 系 feature として使う位置づけ。

生成物は `kaggle/output/train_v3/artifacts/` に取得済み。runtime は 10,111.57 秒。
