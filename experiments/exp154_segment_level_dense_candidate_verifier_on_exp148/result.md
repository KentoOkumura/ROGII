# exp154_segment_level_dense_candidate_verifier_on_exp148 結果

## 状態

Kaggle train v1 と inference v1 / submit 完了。train-side では RMSE -0.029002 改善したが、Public LB は 8.078 で exp148 の 7.960 から +0.118 悪化したため採用しない。

## 仮説

exp135 / exp151 では `tvt_dense` 系候補の全 row feature 化や単純 gate は失敗した。一方、PF worst50 / common PF+ML worst には dense 候補の救済 headroom が残る。現 ML route submitted anchor の exp148 `lgb_mean` を base にし、near-row と path continuity を guard した well/segment-level verifier だけに限定すれば、低頻度に dense 候補を使えるかを診断できる。

## 評価方針

LightGBM の新規学習は行わない。保存済み exp148 `lgb_mean` OOF、exp073 reference OOF、exp072 PF/Beam/dense feature cache を固定入力にして、target-free な segment verifier 条件だけで posthoc 予測を作る。

比較基準:

- 主 baseline: exp148 `lgb_mean` CV 8.501281182 / Public LB 7.960
- historical baseline: exp092 `lgb1` CV 9.322479896 / Public LB 8.350
- reference: exp073 `lgb_mean` CV 9.526374749
- dense headroom: `tvt_dense` / `tvt_densew` / `tvt_dense50` single candidate と oracle readout

## 結果

Kaggle train v1 は `kentookumura/exp154-segment-dense-verifier-train` で完了した。output は `experiments/exp154_segment_level_dense_candidate_verifier_on_exp148/kaggle/output/train_v1`。

| variant | RMSE | delta vs exp148 | within10 | gate rate | max well regression |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base_exp148_lgb_mean` | 8.501281 | 0.000000 | 0.856332 | - | 0.000000 |
| `verifier_dense50_tail1500_q90_min80_clip10_a025` | 8.472280 | -0.029002 | 0.855105 | 0.052558 | +2.287373 |
| `verifier_densew_tail1500_q90_min80_clip10_a025` | 8.472501 | -0.028780 | 0.855105 | 0.052558 | +2.287373 |
| `verifier_dense50_tail1000_q80_min120_clip15_a035` | 8.559563 | +0.058282 | 0.848812 | 0.119929 | +4.831924 |
| `verifier_densew_tail1000_q80_min120_clip15_a035` | 8.564863 | +0.063582 | 0.848591 | 0.119929 | +4.831924 |

best は `verifier_dense50_tail1500_q90_min80_clip10_a025`。exp148 base から RMSE -0.029002 改善した。within10 は 0.856332 から 0.855105 へ -0.001227 悪化した。

worst regime では改善した。

| set | base RMSE | best RMSE | delta | within10 delta direction |
| --- | ---: | ---: | ---: | --- |
| PF `likpf_mean` worst50 | 20.486822 | 20.006159 | -0.480663 | 改善 |
| common PF+ML worst26 | 26.257620 | 25.639148 | -0.618472 | 改善 |
| exp148 worst50 | 23.144352 | 22.852317 | -0.292035 | ほぼ同等 |

bucket guard:

- near `000_050`: delta 0.000000。near guard により変更なし。
- `1000_plus`: delta -0.035276。
- `1000_plus + pf_dense_diff_q4`: delta -0.089011。
- `250_500`: delta +0.014269。
- `500_1000`: delta +0.026098。

by-well では 75 wells 改善、90 wells 悪化。最大悪化は `071d7b45` +2.287373、次に `896d15b9` +1.958276、`13ce113d` +1.798922。最大改善は `b04b58a3` -2.042512、`1b1eba53` -1.915153、`ba48188d` -1.856352。

raw-test parity checklist は required columns present、target-free gate conditions、no LightGBM training が pass。

## Inference / Submission

exp148 saved boosters で current-test base prediction を再生成し、best verifier `verifier_dense50_tail1500_q90_min80_clip10_a025` を適用して Kaggle inference v1 を実行した。

- inference kernel: `kentookumura/exp154-segment-dense-verifier-inference` v1
- output: `experiments/exp154_segment_level_dense_candidate_verifier_on_exp148/kaggle/output/inference_v1`
- rows / wells: 14,151 / 3
- fallback rows: 0
- changed rows vs exp148: 951 (`6.720%`)
- changed wells vs exp148: 1
- abs delta vs exp148: mean 0.168009、p95 2.5、max 2.5
- submission SHA256: `fb6a2be0eb9082974f23806690ffea7552215717b1f6a83d406d6f0da2db1d54`
- submit-check: PASS

Submission ref `54142393` は COMPLETE。Public LB は 8.078。exp148 Public LB 7.960 より +0.118 悪化した。

## 解釈

exp148 anchor に対して、dense candidate を低頻度 segment verifier として使う方向は train-side では支持された。exp135 の broad gate と違い、near rows を壊さず、PF worst50 / common worst26 / longtail high-disagreement を改善した。

ただし within10 が小幅に悪化し、最大 well regression +2.287373 と 90 worsened wells が残っていた。実際の Public LB でも exp148 から悪化したため、train-side の局所改善は hidden/test 側に移らなかったと見る。current-test では 1 well / 951 rows だけが変更され、hard replacement の影響が少数 well に集中した可能性が高い。

## 次

- exp154 は採用しない。ML route submitted anchor は exp148 のまま。
- dense candidate を使う場合は、current-test quantile による hard replacement ではなく、mid bucket regression と top regression wells を明示 guard する。
- 次に進めるなら、exp115 hidden-like stress 相当の readout を先に通し、1 well に変更が集中する gate を避ける。
