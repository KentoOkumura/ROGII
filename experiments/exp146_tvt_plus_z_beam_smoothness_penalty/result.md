# exp146_tvt_plus_z_beam_smoothness_penalty 結果

## 状態

- Kaggle train v1 完了。
- Kernel: `kentookumura/exp146-tvt-z-beam-smooth-train`
- Output: `experiments/exp146_tvt_plus_z_beam_smoothness_penalty/kaggle/output/train_v1`
- 状態: `completed_train_side_beam_improved_not_adopted_no_submit`
- 推論 port / 提出はしない。

## 主要結果

この実験の仮説検証としての主比較対象は、従来の exp072 `beam_mean`。`likpf_mean` は「PF/Beam route の候補として採用できるか」を見るための採用ガードとして別に扱う。

| candidate | RMSE | MAE | within10 | delta vs beam_mean |
| --- | ---: | ---: | ---: | ---: |
| `tvt_plus_z_uslope_c100_uabs005` | 15.566811180 | 10.758215973 | 0.597455754 | -0.207515852 |
| `tvt_plus_z_uslope_c050` | 15.704928817 | 10.759848694 | 0.603956301 | -0.069398216 |
| `beam_mean` | 15.774327032 | 10.898586486 | 0.591649183 | 0.000000000 |
| `tvt_plus_z_uslope_c100_ucurve025` | 15.890491136 | 11.035294747 | 0.584043981 | +0.116164104 |
| `beam_replay_cons` | 16.440420376 | 11.103331481 | 0.595926679 | +0.666093344 |
| `beam_replay_sm5` | 16.521597941 | 11.182477572 | 0.590997490 | +0.747270909 |

best generated Beam variant は `tvt_plus_z_uslope_c100_uabs005`。従来 `beam_mean` に対して RMSE -0.207515852、MAE -0.140370513、within10 +0.005806571 の小幅改善。したがって「Beam mean の改良」という仮説だけを見ると、exp146 は小幅に positive。

## 採用ガード

一方で、既存の `likpf_mean` は RMSE 11.594897672 / MAE 7.067632584 / within10 0.772807479。best TVT+Z Beam は `likpf_mean` から RMSE +3.971913508 悪いため、PF/Beam route の実用候補としては採用しない。

## セグメント所見

- `near_000_050`: best generated `tvt_plus_z_uslope_c100_ucurve025` は 1.021850418。`beam_mean` 1.109429382 から -0.087578964 改善し、`likpf_mean` 1.188877518 からも -0.167027100 改善。ただし `pf_ancc` 0.452880405 の方が強い。
- `longtail_1000_plus`: best generated `tvt_plus_z_uslope_c100_uabs005` は 16.920213787。`beam_mean` 17.132810666 から -0.212596879 改善するが、`likpf_mean` 12.702990216 には +4.217223571 悪い。
- `beam_likpf_gap_top_quartile`: best generated `tvt_plus_z_uslope_c100_uabs005` は 24.058046366。`beam_mean` 25.153127971 から -1.095081605 改善するが、`likpf_mean` 15.582631565 には +8.475414801 悪い。
- representative wells では `91b301ce` は best generated が `beam_mean` / `likpf_mean` の両方に近いが、`ba48188d`、`fef8af96`、`1b1eba53` は実用候補として弱い。

## 判断

`tvt_plus_z_beam_smoothness_penalty` は従来 `beam_mean` の改良としては小幅 positive。ただし `likpf_mean` との差が大きく、inference port / submit 候補としては不採用。

次に扱うなら、Beam 系を直接置換するのではなく、`beam_mean` より少し良くなる TVT+Z Beam variant を confidence feature / segment verifier の材料として使う方向に限定する。
