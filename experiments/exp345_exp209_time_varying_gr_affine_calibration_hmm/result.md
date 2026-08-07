# exp345_exp209_time_varying_gr_affine_calibration_hmm 結果

## 状態

Kaggle CPU canonical kernel version 2でStage 0 fullを完了した。technical gateはPASSしたが、事前定義したscientific AND gateをFAILしたため、判定は`stage_failed_close_without_rescue`、実験状態は`stage_0_full_failed_closed`である。Stage 1、inference、submission、post-hoc parameter/grid救済は実行しない。

## 仮説

exp209の遷移・観測分散・state grammarを固定し、frozen exp209 base pathからcurrent-well causal filterで一度だけ推定したGR affine `a_t,b_t` scheduleを観測中心へ適用すれば、固定identity観測よりsuffixの緩やかなscale/offset driftへ適応できると仮定した。

## 変更点と実行量

- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Route: `pf_beam`
- 単一変更: GR observation centerをidentity `a=1,b=0`から凍結済み`a_t,b_t` scheduleへ置換
- Stage 0: last-640 prefix mask、773 wells、親773 + variant773 = 1,546 HMM runs
- prediction: 494,720 rows
- LightGBM config / 学習fold / booster / PF / Beam / GPU: 0 / 0 / 0 / 0 / 0 / 0
- Kaggle kernel: `kentookumura/exp345-gr-affine-hmm-runtime-microbenchmark` version 2、CPU、internet off

exp209 zero-fill std `sigma_GR`、欠損row weight 1、GR補間、Gaussian emission、41 rate states、`sig_r=0.002`、`sig_p=0.02`、position floor、momentum、prior、posterior meanは固定した。

## Technical gate

| 項目 | 実績 | 条件 | 判定 |
| --- | ---: | ---: | --- |
| prediction rows / wells | 494,720 / 773 | 494,720 / 773 | PASS |
| HMM runs | 1,546 | 1,546 | PASS |
| parent / variant runs | 773 / 773 | 773 / 773 | PASS |
| finite prediction | 全行finite | 必須 | PASS |
| fallback率 | 0.0% | `<=50%` | PASS |
| posterior正規化最大絶対誤差 | `3.219646771412954e-15` | `<=1e-6` | PASS |
| runtime | 4,871.0129秒（1.3531時間） | `<=8.5時間` | PASS |

technical gateは全項目を満たした。

## Scientific gate

### RMSE

| scope | 親RMSE | 候補RMSE | 改善量（ft） |
| --- | ---: | ---: | ---: |
| overall | 14.501048 | 14.331543 | +0.169505 |
| fold 0 | 25.036652 | 24.968952 | +0.067700 |
| fold 1 | 7.496659 | 7.496915 | -0.000256 |
| fold 2 | 11.246859 | 11.234554 | +0.012305 |
| fold 3 | 12.530363 | 11.928668 | +0.601695 |
| fold 4 | 9.149097 | 8.807567 | +0.341530 |

### Gate内訳

| 項目 | 実績 | 条件 | 判定 |
| --- | ---: | ---: | --- |
| overall改善 | +0.169505 ft | `>=0.05 ft` | PASS |
| 改善fold | 4/5 | `>=4/5` | PASS |
| GR predictive NLL | identity 4.651670 → affine 4.646152 | 改善 | PASS |
| boundary jump p95 | 0.010089 sigma | `<=3 sigma` | PASS |
| hidden-like | 2 scopeともreadoutなし | 2 scope非悪化 | FAIL |
| worst well delta | +9.354827 ft | `<=+0.25 ft` | FAIL |

by-wellでは373/773 wellsが改善し、400/773 wellsが悪化した。best well `0dc835f3`は`-17.880884 ft`、worst well `c03b9305`は`+9.354827 ft`だった。candidate RMSEのp95とparent RMSEのp95の差は`-0.037121 ft`だが、worst-well guardとhidden-like evidence completenessを満たさないため、scientific AND gateはFAILである。

## 再現性証拠

- Stage 0 promotion gate raw SHA256: `39296d1b900463c27f1fd65fbaa265e3c1a3a6b9d42afd9322eb03ac6140525a`
- input manifest raw SHA256: `fc81201b445b86561c851d4e4c8fc8612d852652f444fb2733d76d507ff31de9`
- scientific contract declared SHA256: `0814b2b8788204fc3561bbe37c7dc64b46f79bad353865f300272a7d6cc73b47`
- freeze manifest SHA256: `1f4aed295a28a744de92a076ff1c21babf8841f947966a7c98f35d0f854c3509`
- identity content SHA256: `7c1e76c38b542db07890adf3e7cea52f9a8db23d1ee4bb8e63e588741003bb84`
- prediction raw / decompressed SHA256: `8e038204dc5768dc68c77931f72bddf883f1afe67ed3461119892e80800467d0` / `f2ff65b78a66c88e9993f2c362fbd9db445061670980cfffccf449ef81d4bfbc`
- affine schedule raw / decompressed SHA256: `0470f6ee70d91eb9a7501f5b58c6c4e4de89e1eab0b9b1b6a94eb5342c8c10b9` / `51827246e6b7154ff39d3d6a8c07d1bd0dd43715090b9f11036b67960d9bf0f0`
- model / submission SHA: 非該当

gzip 2生成物の展開後SHAはKaggle summary記録とローカル再計算で一致した。promotion gate JSONとpaired metrics CSVだけを実験配下`artifacts/`へ保存し、大きなoutputは一時領域で検証した。

## 考察

全体RMSE、4/5 folds、GR NLL、boundaryは支持されたため、affine scheduleが一部wellのscale/offset driftを補正する信号はある。一方、改善はwell横断で安定せず、悪化wellが400/773と過半数で、最悪tailは許容値の約37倍に達した。少数の大幅改善がpooled RMSEを押し下げても、未知wellで安全に適用できる候補ではない。

また、Stage 0 paired metricsには事前必須のhidden-like spatial / typewell-purged scopeが生成されず、非悪化を証明できなかった。証拠欠落をPASS扱いせずfail-closedにした。

32-well microbenchmark previewはoverall`-0.220170 ft`、1/5 folds、12/32 wellsでnegativeだった。full Stage 0ではpooled値が改善へ反転したが、per-well不安定性とworst tailという警告は残った。

## 結論と次のアクション

事前契約どおり`stage_failed_close_without_rescue`とし、このfamilyを閉じる。Stage 1を実行せず、affine/process-noise/grid、transition、sigma、missing weight、blendによる同一実験内救済も行わない。実装済み項目は`KAGGLE_DIRECTION.md`のアイデアバックログから削除する。再訪する場合は、独立した新しい根拠、別実験の事前設計、ユーザー確認を必要とする。
