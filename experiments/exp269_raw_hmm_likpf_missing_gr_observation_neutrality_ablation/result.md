# exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation 結果

## 結論

Kaggle CPU Stage 1 version 1を完了した。raw GR欠損rowのGR emissionをstate-neutralにすると、exp209 raw typewell-GR exact HMM controlに対してoverall RMSEが`+1.410212 ft`悪化した。事前固定guardは不通過で、判断は`stage1_fail_pf_closed`とする。

likelihood-PF Stage 2、raw-test inference、submissionは実行しない。

## 比較契約

- control: exp209/exp205 raw typewell-GR exact HMM保存cache。再生成なし。
- variant: raw GR missing evaluation rowだけGR emissionを全state `0`にする。
- 固定: GR補間、grid、41 rate states、transition、initialization、sigma、score rows。
- 不使用: exp223 self-GR、LightGBM unary、run-length gate、likelihood-PF。
- route: `pf_beam`。

control decompressed SHAは事前固定値`8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`と一致した。

## 主要結果

| 評価面 | control RMSE | neutral RMSE | delta (neutral - control) | guard |
|---|---:|---:|---:|---|
| overall | 11.938287 | 13.348499 | +1.410212 | FAIL (`<= -0.02`) |
| raw GR missing | 11.948064 | 14.496321 | +2.548257 | FAIL (`<= 0`) |
| raw GR observed | 11.933740 | 12.779855 | +0.846115 | FAIL (`<= +0.02`) |
| distance 1000+ | 13.135431 | 14.719236 | +1.583805 | FAIL (`<= +0.02`) |
| hidden-like spatial | 12.564491 | 16.027490 | +3.462999 | FAIL (`<= +0.02`) |
| hidden-like typewell-purged | 12.367244 | 15.923789 | +3.556545 | FAIL (`<= +0.02`) |
| focus well `11d0f5ac` | 21.160939 | 21.514271 | +0.353332 | readout |

- 対象は3,783,989 rows / 773 wells。raw GR missingは1,200,837 rowsで、全773 wellsに存在した。
- predictionは3,783,349 rows / 全773 wellsで変化した。平均絶対差は2.298263 ft、最大絶対差は80.893875 ft。
- worst wellは`e03b45fd`で、RMSE 10.657176 -> 61.824631、`+51.167455 ft`。missing fractionは0.446462、longest missing runは56 rowsだった。
- posterior std meanは2.940914 -> 3.671249へ増え、欠損rowだけでは3.271540 -> 4.413686へ増えた。

## Stage 1 guard

10条件中、数値安全性3条件だけが通過した。

- PASS: prediction finite coverage 100%、std finite coverage 100%、ID mismatch 0。
- FAIL: overall、missing、observed、1000+、hidden-like 2面、worst-well。
- `pf_stage_eligible=false`、`pf_stage_executed=false`。

## 解釈

仮説は棄却された。補間GRのemissionを欠損rowで完全に外すと、欠損row自身だけでなくHMM smoothingを通じてobserved rowにも変化が伝播した。短いmissing runの主群でも悪化しており、`1_4` rowsは`+1.958686 ft`、`5_31` rowsは`+4.222560 ft`だった。欠損rowにおける補間GR emissionは、少なくとも現行exp209 grammarでは有害な疑似観測ではなく、pathを安定させる重要な拘束として働いている。

一部の局所bucketには改善があるが、事後gateやrun-length ruleを作る根拠には使わない。blanket neutralityのPF移植、sigma/temperature救済、mask gridは閉じる。次のraw-HMM候補監査はmissing-GR処理を固定した既存`exp270_exact_hmm_posterior_mode_candidate_audit`で行い、本branchの救済とは扱わない。

## 実行・再現性

- Kaggle kernel: `kentookumura/exp269-raw-hmm-missing-gr-neutrality-train` version 1、id_no `127592556`。
- runtime: Kaggle CPU、19,573.731秒（約5時間26分14秒）、outer workers 2、Numba threads 2。
- Python 3.12.13、NumPy 2.0.2、pandas 2.3.3、GPU/internet false。
- variant prediction decompressed SHA: `4dfcceccccb1496e89601566019f0dd8f649cb4c5711f1d9a2f83be617a39976`。
- row audit decompressed SHA: `818126ff9f538429befefa2890e4e0c7ef9a03d6255f00634336630dc941c88b`。
- 詳細なgroup、by-well、finite coverage、input/output SHAは`metrics.json`とKaggle成果物summaryに記録した。
