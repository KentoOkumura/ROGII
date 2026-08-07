# exp209_exp072_exp205_joint_exact_parity_fast_cache_generation 結果

## 仮説

exp072 full replay feature cache と exp205 exact HMM cache / direct comparison を同一 notebook 内で扱い、exp072 DataFrame を in-memory で comparison に渡す。さらに HMM を well 外側で安全に並列化すれば、PF/HMM の値を変えずに wall time を短縮できる。

## 設定

- 親: `exp072_exp063_full_replay_feature_cache`
- 親: `exp205_exact_hmm_smoother_exp072_compatible_cache_audit`
- Route: `pf_beam`
- 検証: train-side joint generation parity / runtime audit
- メトリック: wall time、feature content SHA、exp205 direct comparison RMSE parity
- シード: exp072 stable SHA256 per-well seed、HMM no RNG
- 採用 HMM runtime: v5 `outer_workers=2`, `numba_num_threads=2`
- 追加評価: v6 `outer_workers=4`, `numba_num_threads=1`

## 結果

Kaggle train v5 は完了。`outer_workers=2`, `numba_num_threads=2` により HMM wall time は v4 の `19,749.099 sec` から `11,285.868 sec` へ短縮した。全体 wall time も `20,203.290 sec` (約 5h36m43s) で、v4 より `12,580.970 sec` 速く、当初の 6h 未満 target を達成した。

追加で v6 `outer_workers=4`, `numba_num_threads=1` も完了したが、total `28,768.406 sec`、HMM `14,627.100 sec` で v5 より遅かった。HMM decompressed SHA と RMSE 近似一致は保てたため correctness 面の問題ではなく、parallelism 配分の効率が v5 より悪いと判断する。

HMM feature cache は exp205 v2 と decompressed SHA が完全一致し、best comparison candidate も exp205 v2 と同じ `blend_likpf_hmm_w500`。RMSE 差 `3.8106e-06` は元の strict tolerance `1e-9` を超えるが、ユーザー確認により近似 RMSE 一致として許容する。一方、exp072 full replay cache は raw gzip SHA / decompressed SHA とも exp072 v2 reference に一致しておらず、full artifact exact parity は未証明。

| メトリック | 値 |
| --- | --- |
| Kaggle train status | v6 complete / selected runtime v5 |
| wall time | 20,203.290 sec (約 5h36m43s) |
| v4 からの短縮 | -12,580.970 sec (約 -3h29m41s) |
| exp072 elapsed | 8,723.765 sec |
| HMM elapsed | 11,285.868 sec |
| HMM v4 からの短縮 | -8,463.231 sec (約 -2h21m03s) |
| rows / wells | 3,783,989 / 773 |
| best comparison candidate | `blend_likpf_hmm_w500` |
| best comparison RMSE | 10.269696146642758 |
| exp072 `likpf_mean` RMSE | 11.594897672217703 |
| best delta vs exp072 `likpf_mean` | -1.3252015255749452 |
| exp072 reference parity | FAIL |
| exp205 HMM reference parity | PASS |
| metric parity | candidate PASS、RMSE 近似一致 ACCEPT |
| HMM outer workers | 2 |
| Numba threads requested / effective | `2` / `2` |

Runtime comparison:

| run | outer / numba | total elapsed | HMM elapsed | v5 比 | 判定 |
| --- | --- | ---: | ---: | ---: | --- |
| v4 | `1` / `4` | 32,784.260 sec | 19,749.099 sec | +12,580.970 sec | 不採用 |
| v5 | `2` / `2` | 20,203.290 sec | 11,285.868 sec | 0.000 sec | 採用 |
| v6 | `4` / `1` | 28,768.406 sec | 14,627.100 sec | +8,565.116 sec | 不採用 |

基準の単純合算は exp072 v2 `17,728.972 sec` + exp205 v2 HMM `15,041.783 sec` = `32,770.755 sec`。exp209 v5 は `20,203.290 sec` で、単純合算より `12,567.465 sec` 速い。

HMM の well ごとの平均 elapsed は v5 で `28.9516 sec` と v4 の `25.3525 sec` より大きいが、これは 2 wells を同時処理して CPU を分け合うためで、採否は wall time で判断する。wall time では HMM が v4 から約 2.35 時間短縮している。

SHA:

| 対象 | generated | reference | 判定 |
| --- | --- | --- | --- |
| exp072 raw gzip | `cff5e56193100a8dbc2b28471b7a75404f99deb1fa6bcb1d4116f473289606a7` | `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18` | FAIL |
| exp072 decompressed | `0503de0512302b06309d26e09fc06ba5095db0ef4d610b1508afe8c8d07ca536` | `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350` | FAIL |
| HMM raw gzip | `8957e92f3e010f307ab0918316060a14e7479d5aba8225676b560272728442ba` | `ca5343ca04b3774fcc4bfb95c96ba1f43a9a9ac70202e545019b3dba308b87d6` | gzip timestamp 等のため不一致 |
| HMM decompressed | `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5` | `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5` | PASS |

## Kaggle 実行

- kernel: `kentookumura/exp209-joint-exact-parity-train`
- selected version: 5
- latest evaluated version: 6
- URL: `https://www.kaggle.com/code/kentookumura/exp209-joint-exact-parity-train`
- runtime: CPU, internet off
- status確認: `KernelWorkerStatus.COMPLETE`
- selected small output: `/tmp/kaggle-output/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/train_v5_small`
- v6 small output: `/tmp/kaggle-output/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/train_v6_small`

## 再現性

- deterministic anchor: いいえ。train-side feature generation audit。
- seed policy: exp072 stable SHA256 per-well、HMM no RNG。
- kernel version: 6 complete。採用 runtime は version 5。
- feature content SHA: HMM decompressed SHA は exp205 v2 と一致。exp072 full cache SHA は不一致。
- model SHA / manifest SHA: 対象外。
- prediction SHA: 対象外。
- submission SHA: 対象外。
- rerun result: v1/v2 early failure fixed; v3/v4/v5/v6 complete。metric parity は近似一致として許容、HMM feature parity は PASS、runtime target は v5 で達成。v6 は v5 より遅いため不採用。

## 解釈

`outer_workers=2`, `numba_num_threads=2` は有効だった。HMM の exact decompressed feature parity と best RMSE 近似一致を保ったまま、全体 runtime は 5.61h まで短縮した。`outer_workers=4`, `numba_num_threads=1` は HMM per-well elapsed が `74.9061 sec` まで増え、wall time も v5 から悪化した。

残る注意点は、exp072 full cache の再生成 artifact が v2 reference と SHA 一致しないこと。今回の判断基準では RMSE 近似一致を採用しているため、exp209 は runtime target 達成として完了扱いにする。

## 次

現 exp209 の既定は v5 の `outer_workers=2`, `numba_num_threads=2` とする。さらなる短縮は、外側並列の追加探索より exact-full-cache 条件を外した likPF-only/slim cache 化の方が見込みが高い。
