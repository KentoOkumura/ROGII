# exp205_exact_hmm_smoother_exp072_compatible_cache_audit 結果

## 仮説

exact HMM smoother の posterior mean/std が exp072 `likpf_mean` と補完的なら、HMM 単体または固定 blend が train-side direct comparison で改善する可能性がある。

## 設定

- 親: `exact_hmm_smoother_exp072_compatible_cache_audit` backlog
- 比較基準: `exp072_exp063_full_replay_feature_cache`
- 検証: train-only feature cache direct audit
- メトリック: RMSE TVT
- シード: HMM default 生成は乱数なし
- HMM: `step=0.35`, `n_rates=41`, `rate_span=0.10`, `sig_r=0.002`, `sig_p=0.02`, `mom=0.998`

## 結果

Kaggle train v2 は完走した。HMM feature cache は 773 wells / 3,783,989 rows を全件生成し、exp072 cache との `id` mismatch は 0。

| 候補 | RMSE | MAE | within10 | exp072 likpf_mean 比 RMSE |
| --- | ---: | ---: | ---: | ---: |
| `blend_likpf_hmm_w500` | 10.269700 | 6.399213 | 0.793202 | -1.325198 |
| `blend_likpf_hmm_w250` | 10.568837 | 6.587048 | 0.785897 | -1.026061 |
| `blend_likpf_hmm_w750` | 10.758297 | 6.428405 | 0.795246 | -0.836601 |
| `exp072_likpf_mean` | 11.594898 | 7.067633 | 0.772802 | 0.000000 |
| `hmm_mean_tvt` | 11.938297 | 6.769557 | 0.784379 | +0.343399 |

距離 bucket では best blend が全 bucket で `exp072_likpf_mean` を改善した。`1000_plus` は 12.702990 -> 11.282505、delta -1.420485。near も `000_050` で 1.178524 -> 0.830660、delta -0.347865。

well 単位では `blend_likpf_hmm_w500` が 539 wells 改善、234 wells 悪化。最大悪化は `b19b0395` の +23.036816 RMSE、最大改善は `86454a6f` の -28.133145 RMSE。HMM 単体は 451 wells 改善、322 wells 悪化で、最大悪化 +48.316191 RMSE が残る。

`hmm_std` と HMM absolute error の相関は 0.399484。posterior std は粗い risk signal にはなるが、最低 bin が完全な単調増加から外れるため、そのまま hard gate には使わない。

| メトリック | 値 |
| --- | --- |
| CV | 10.269699957 (`blend_likpf_hmm_w500`) |
| Public LB | - |
| Private LB | - |

## 再現性

- deterministic anchor: false
- seed policy: default HMM generation は no RNG
- kernel version: Kaggle train v2
- feature content SHA: `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`
- generated feature gzip SHA: `ca5343ca04b3774fcc4bfb95c96ba1f43a9a9ac70202e545019b3dba308b87d6`
- model SHA / manifest SHA: not applicable
- prediction SHA: not applicable
- submission SHA: not applicable
- rerun result: v1 は kernelspec metadata 不足で起動前 error。v2 は同一 kernel id で完走。

## 解釈

train-side direct comparison では HMM と exp072 `likpf_mean` の補完性が強く、固定 50/50 blend は全体 RMSE、MAE、within10、全距離 bucket で改善した。step delta も `exp072_likpf_mean` より滑らかで、`abs_step_delta_mean` は 0.029514 -> 0.017669、`|delta| > 0.1` rate は 0.047355 -> 0.012700。

一方で HMM 単体は `likpf_mean` より RMSE が悪く、best blend でも worst-well regression が +23ft 級残る。raw-test-compatible regeneration、hidden-like stress、worst-well guard が未確認なので、この実験から直接 inference / submit へは進めない。

## 次

`exact_hmm_likpf_blend_raw_test_port_guard` として、raw-test-safe HMM regeneration、hidden-like guard、worst-well guard を確認する follow-up に切り出す。exp205 自体は train-only audit として完了し、提出はしない。
