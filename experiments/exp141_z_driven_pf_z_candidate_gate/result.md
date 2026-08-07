# exp141_z_driven_pf_z_candidate_gate 結果

## 仮説

`pf_z` は direct replacement では弱いが、Z trajectory と同期する低頻度区間だけなら `likpf_mean` の補助候補として改善する可能性がある。

## 設定

- 親: `exp072_exp063_full_replay_feature_cache`
- 検証: train pseudo-tail feature cache 上の posthoc gate audit
- メトリック: RMSE
- シード: 42。ただし exp141 自体に新規乱数はない。

## 結果

- Kaggle kernel: `kentookumura/exp141-z-pfz-gate-train` v1
- 状態: COMPLETE
- 行数 / well 数: 3,783,989 rows / 773 wells
- baseline `likpf_mean`: RMSE 11.594897672、MAE 7.067632675、within10 0.772807479
- `single_pf_z`: RMSE 17.788171172、baseline から +6.193273499 悪化
- `single_pf_ancc`: RMSE 14.493050690、baseline から +2.898153017 悪化
- `single_beam_mean`: RMSE 15.774327032、baseline から +4.179429360 悪化
- oracle `likpf_mean` + `pf_z`: RMSE 9.115200716、baseline から -2.479696957 改善
- oracle core PF/Beam: RMSE 6.953036836、baseline から -4.641860836 改善

設定した target-free gate はすべて `likpf_mean` baseline より悪化した。

| variant | RMSE | delta vs `likpf_mean` | gate rate | max well regression |
| --- | ---: | ---: | ---: | ---: |
| `base_likpf_mean` | 11.594897672 | 0.000000000 | 0.000000 | 0.0000 |
| `seg_zq75_alignq60_diffq70_sr010_min32_clip20_a050` | 11.633719432 | +0.038821760 | 0.084827 | +4.8418 |
| `seg_zq80_alignq65_diffq65_sr005_min24_clip20_a075` | 11.680175213 | +0.085277540 | 0.050000 | +8.5004 |
| `row_zq90_alignq75_diffq75_sr001_clip25` | 11.714344701 | +0.119447029 | 0.010000 | +11.0704 |
| `seg_zq85_alignq70_diffq70_sr003_min16_clip25` | 11.829114782 | +0.234217109 | 0.030000 | +12.7026 |
| `row_zq85_alignq70_diffq70_sr003_clip25` | 11.866666159 | +0.271768487 | 0.030000 | +13.8494 |
| `well_zq90_alignq75_sr005_tail500_clip25` | 12.398925507 | +0.804027835 | 0.049974 | +22.6906 |

代表 Z-driven well では、`ba48188d` と `fef8af96` で `single_pf_z` が `likpf_mean` より大きく良い一方、`91b301ce` では `pf_z` が悪化し `pf_ancc` / oracle core のほうが良い。global な target-free gate はこの差を十分に選別できず、row-wise gate では step >= 10 / 25 の不連続も増えた。

## 再現性

- deterministic anchor: false
- seed policy: `no_new_rng_posthoc_saved_cache_audit`
- kernel version: `kentookumura/exp141-z-pfz-gate-train` v1
- feature content SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- raw feature file SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- feature schema SHA: `700d38149f583c3ab6574ea7b163c3c8709c2514b675bea381d822f82f4809b8`
- model SHA / manifest SHA: 新規モデルなし
- prediction SHA: submission prediction なし
- submission SHA: submission なし
- rerun result: 未実行

## 解釈

`pf_z` は一部 well / row では救済 headroom を持つが、今回の `abs(dZ/dMD)`、Z slope alignment、candidate disagreement、roughness、segment length を使った target-free gate では選択精度が足りなかった。`likpf_mean` + `pf_z` oracle は大きく改善するため候補集合の余地は残るが、hard switch のまま推論化する根拠はない。

この実験は train-side rejected とし、raw-test inference port と submission は行わない。`pf_z` を使う場合は、直接 gate ではなく segment-level verifier、candidate confidence feature、または `z_slope_posthoc_correction_on_pfbeam_candidates` のような小補正方向に限定する。

## 次

1. `z_driven_pf_z_candidate_gate` backlog は完了 / 不採用として閉じる。
2. `pf_z` hard switch は提出候補にしない。
3. oracle headroom は後続の segment-level verifier / confidence feature の参考値としてのみ残す。
