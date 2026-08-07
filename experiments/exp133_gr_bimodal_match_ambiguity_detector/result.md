# exp133_gr_bimodal_match_ambiguity_detector 結果

## 仮説

GR matching の score curve が +/-15-25ft decoy や複数 peak を持つ row では、hard mode selection が危険になる。真値側 mode を当てに行くのではなく、ambiguity / uncertainty / midpoint proxy / near-row guard の材料として使う。

## 設定

- 親: `gr_bimodal_match_ambiguity_detector` backlog
- 入力: exp072 feature cache、exp073 OOF、exp092 OOF、raw train horizontal GR / TVT_input
- 検証: train-side diagnostic
- メトリック: ambiguity flag 別 RMSE / MAE / within10、mode commit vs midpoint proxy、candidate metrics
- シード: 42。ただし本実験に乱数処理なし。

## 結果

Kaggle train v2 完了。LightGBM 学習、推論、提出はなし。

| メトリック | 値 |
| --- | --- |
| rows / wells | 3,783,989 / 773 |
| runtime | 1783.534 sec |
| best reference | `pred_exp092_lgb1` |
| best RMSE | 9.322479895503927 |
| best MAE | 5.980980396270752 |
| best within10 | 0.8220470514052762 |
| ambiguous rate | 0.5668565630912781 |
| flat score rate | 0.4328453242778778 |
| mean ambiguity score | 0.48863860964775085 |
| feature count | 40 |
| Public LB | - |
| Private LB | - |

候補別の主な結果:

| candidate | RMSE | MAE | within10 |
| --- | ---: | ---: | ---: |
| `pred_exp092_lgb1` | 9.322480 | 5.980980 | 0.822047 |
| `pred_exp092_lgb_mean` | 9.343064 | 5.961740 | 0.823412 |
| `pred_exp073_lgb_mean` | 9.526375 | 6.159766 | 0.813495 |
| `pred_likpf_mean` | 11.594898 | 7.067633 | 0.772807 |
| `grbm_likpf_midpoint_blend` | 21.681648 | 11.037802 | 0.653901 |
| `grbm_midpoint_proxy` | 66.623786 | 31.178198 | 0.342099 |
| `grbm_mode_commit_proxy` | 73.891128 | 31.907003 | 0.364051 |

Ambiguity bucket:

| bucket | rows | exp092 lgb1 RMSE | exp073 RMSE | likPF RMSE | mode commit RMSE | midpoint RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ambiguous=0 | 1,639,010 | 9.422932 | 9.675804 | 11.889749 | 88.103030 | 79.084000 |
| ambiguous=1 | 2,144,979 | 9.244988 | 9.410595 | 11.364443 | 60.833538 | 55.239842 |

Flat score bucket:

| bucket | rows | exp092 lgb1 RMSE | exp073 RMSE | likPF RMSE | mode commit RMSE | midpoint RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flat=0 | 2,146,107 | 9.243953 | 9.409597 | 11.364516 | 60.897225 | 55.310439 |
| flat=1 | 1,637,882 | 9.424383 | 9.677255 | 11.890011 | 88.061245 | 79.033275 |

## 再現性

- deterministic anchor: false
- seed policy: no_new_rng_gr_ambiguity_diagnostic
- kernel version: `kentookumura/exp133-gr-bimodal-ambiguity-train` v2
- feature content SHA: raw gzip `cb0e9af9b55ba941b79c78eb3480f2d78207414edd7e62786e8485e81ed70f26`
- feature content SHA decompressed: `acdb77139e7f73efda4d8fe5dc29799823c4e46a687af5030e70a9dbe0b8d50a`
- feature schema SHA: `4612012736f6936bafc745d4ff3b54b92a128712de89b0a32360f3416d40889c`
- model SHA / manifest SHA: なし
- prediction SHA: なし
- submission SHA: なし
- rerun result: v1 は indexing bug で失敗、v2 で修正して完了

## 解釈

GR score curve の ambiguity / flatness は計算でき、feature cache としては利用可能。ただし、detector の top mode や midpoint を TVT proxy として直接使う方向は明確に壊れる。`grbm_mode_commit_proxy` と `grbm_midpoint_proxy` は RMSE 60-90 級で、`likPF` より大幅に悪い。

また `grbm_ambiguous_flag` は exp092 の悪化領域を単純には示していない。むしろ ambiguous=1 の exp092 RMSE は 9.244988 で ambiguous=0 の 9.422932 より良い。flat=1 は exp092 / exp073 / likPF がやや悪い regime だが、直接 mode commit / midpoint はさらに悪いため、guard や averaging policy には使わない。

結論として、direct correction / candidate path / inference port / submit はしない。残す場合は `grbm_top1_top2_margin`、entropy、flat flag、zero scores、shift gap などを exp092 系 LightGBM の add-only confidence feature として小さく評価する。

## 次

`gr_bimodal_match_ambiguity_detector` は完了として backlog から外す。追加実験を切るなら、直接 proxy ではなく exp092 add-only feature に限定する。
