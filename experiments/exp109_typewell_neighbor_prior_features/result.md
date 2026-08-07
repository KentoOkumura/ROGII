# exp109_typewell_neighbor_prior_features 結果

## 仮説

同じ native typewell overlap group に属する train-fold wells の TVT drift curve は、query well の既存 PF/Beam/likPF 候補の誤差方向を弱く説明できる可能性がある。

## 設定

- 親: `exp099_pf_multi_observation_likelihood_probe`
- 参照: `exp065_typewell_supertype_cluster_cv_audit`
- 検証: well-grouped 5 folds train pseudo-tail OOF neighbor prior audit
- メトリック: RMSE / MAE / within10 / bucket / by-well
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| CV | best `native_overlap_0p999_likpf_mean_corr_a0p2_c40` RMSE 11.143359521 |
| Public LB | - |
| Private LB | - |

Kaggle train v2 完了。rows 3,783,989 / wells 773。比較対象の `likpf_mean` は RMSE 11.594897672 / MAE 7.067632584 / within10 0.772807479。best neighbor prior correction は RMSE 11.143359521 / MAE 7.025321534 / within10 0.779883345 で、`likpf_mean` から RMSE -0.451538151、within10 +0.007075866 改善した。

距離 bucket では best correction が全 bucket で `likpf_mean` より RMSE 改善した。delta は 0-50ft -0.199831、50-100ft -0.269622、100-250ft -0.255874、250-500ft -0.186812、500-1000ft -0.154440、1000ft+ -0.500051。

一方、well 単位では 413 wells 改善 / 345 wells 悪化 / 15 wells 同値。平均 delta は -0.206130、中央値 -0.093055。最大悪化は `f88ddb26` +6.594183 RMSE、最大改善は `7987f2f2` -6.447593 RMSE。global 改善は強いが、worst-well guard なしで提出候補にしない。

## 再現性

- deterministic anchor: false
- seed policy: deterministic well fold assignment with fixed seed 42
- kernel version: `kentookumura/exp109-typewell-neighbor-prior-train` v2
- feature cache SHA: exp099 raw `4bd9df60f5c09f7a3029dac399afef73aa45b0158a7fd06a62a56f85fd0fde38` / decompressed `1939d536b1e56f7c0ea3847cc386ef769b0d33759d16e816c9ce180f0532df9a`
- cluster assignment SHA: `dcda8588cc1dd9261bafae7de00c890393e38b8a0ca0eb86fbba18a2cffc4a50`
- OOF prediction SHA: raw `cc4c017baff6410ce8a0cf52e0a0e76d6f7309fd9b163eea43c0173b7e5fb660` / decompressed `ec1e105a3021badf7441768329e94dfce874f110927bf9af1a3968ddbc609e29`
- model SHA / manifest SHA: model なし
- submission SHA: submission なし
- rerun result: 未実行

## 解釈

native overlap neighbor prior は、少なくとも train pseudo-tail の `likpf_mean` 後段補正として明確に有効。`native_overlap_1` と `native_overlap_0p999` は今回同じ結果になり、prior valid rate は 0.973144、平均 row neighbor count は 19.597。`exact_hash` は prior valid rate 0.029846 と coverage が低く主候補ではない。

ただし、これは PF 内部 likelihood の実装ではなく、既存候補への後段補正。well 単位の悪化が大きいため、次は full train を使う inference-compatible prior 再生成、raw-test feature parity、worst-well / prior std / neighbor count gate を確認する必要がある。

## 次

`typewell_neighbor_prior_rawtest_parity_gate` として、best correction を固定し、visible/full train inference flow で同じ prior を再生成できるか、prior coverage / correction 分布が train OOF とずれないか、worst-well regression を gate できるか確認する。submit はその後。
