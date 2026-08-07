# exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264 結果

## 状態

Kaggle private CPU version 4でStage 0を完了した。technical gateは全PASS、
scientific all-AND gateはFAILし、`stage_0_failed_close_without_rescue`で閉じた。

## 仮説

exp368の連続weak riskは、exp264の既存selector scoreを超えて、
`likpf_mean` bad10 rowで既存scoreが指名したother candidateによる
within10回復を識別できる。

## 設計

- 親: `exp264_exp263_candidate_confidence_dual_selector`
- 補助入力: `exp368_marginalized_reliability_pf`
- Route: `ml_model`
- Stage 0: 0 model / 0 boosterのcandidate-advantage readout
- feature: overlapping exp368 blockの`weak_posterior_mean`算術平均1列
- control: 保存stable within-well circular block shift
- primary domain: `primitive_pair_bank` 11候補
- secondary domain: `primitive_fixed_bank` 7候補
- seed: Stage 0 RNGなし

## 結果

| メトリック | 値 |
| --- | --- |
| Kaggle kernel | version 4 / id_no `128626512` / COMPLETE |
| Stage 0 runtime | 129.300秒 |
| Technical gate | 15/15 PASS |
| Scientific gate | 4/12 PASS・総合FAIL |
| Primary pooled AUC | 0.520214（閾値0.60未満） |
| Primary circular AUC / 差 | 0.523467 / -0.003253 |
| Primary margin-conditional AUC | 0.458846（閾値0.55未満） |
| Primary hidden-like AUC | spatial 0.527468 / typewell-purged 0.513626 |
| Primary Q4-Q1 realized advantage | +3.879372 ft（PASS） |
| Secondary pooled AUC / circular差 | 0.504233 / -0.003988 |
| 専用contract test | 9件PASS |
| Jupytext / py_compile / Ruff / strict validation | PASS |
| CV | - |
| Public LB | - |
| Private LB | - |

## 再現性

- deterministic submission anchor: いいえ
- input SHA: 固定5入力を全一致
- feature content SHA: `b71b7d57...5955ff7`
- selector surface SHA: `986a02c8...a2db83`
- Stage 0 summary SHA: `2b46291b...b145909`
- model / prediction / submission SHA: 非該当
- kernel version: 4 / id_no `128626512`
- downloaded output: `kaggle/output/train_v4`
- actual output SHA: feature / scope metrics / nomination distribution / summaryを照合PASS
- canonical compact self-contained train: 11章
- 親exp264 train source: 7章、465行

## 解釈

weak riskの高quartileほどrealized advantageは大きく、Q4-Q1はprimary
`+3.879372 ft`、secondary `+3.676856 ft`でPASSした。しかし回復可能rowの
順位識別はprimary pooled AUC `0.520214`に留まり、circular controlよりも
`0.003253`低い。margin条件付きとhidden-like 2面も閾値未満であり、exp264へ
1列をadd-onlyする根拠にはならない。exp368のPF branch FAILも維持する。

## 次

事前契約どおり、threshold、反転、bucket、domain、candidate subset、
gateを救済せずbranchを閉じる。Stage 1の40 CPU selector boosters、
downstream TVT、inference、submissionは実装・実行しない。
