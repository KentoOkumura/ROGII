# typewell GR TVT fit audit

- generated_at_utc: `2026-07-05T04:36:34.707982+00:00`
- primary rank: lower grid `z_rmse`, with minimum overlap/coverage filters
- `prefix`: rows with finite `TVT_input`; `hidden`: train rows with missing `TVT_input`; `full`: all finite train `TVT` rows

## Grid summary

| split | view | wells | provided best rate | provided top5 rate | median rank | provided corr median | provided zRMSE median | best zRMSE median | median best-provided zRMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test | prefix | 3 | 0.000 | 0.000 | 12.000 | 0.820 | 0.601 | 0.576 | -0.019 |
| train | full | 773 | 0.076 | 0.365 | 8.000 | 0.831 | 0.582 | 0.573 | 0.000 |
| train | hidden | 773 | 0.010 | 0.061 | 9.000 | 0.787 | 0.652 | 0.669 | 0.000 |
| train | prefix | 773 | 0.079 | 0.369 | 8.000 | 0.829 | 0.585 | 0.575 | 0.000 |

## Exact selected-pair summary

| view | role | wells | coverage median | corr median | zRMSE median | raw RMSE median | raw MAE median |
|---|---|---:|---:|---:|---:|---:|---:|
| full | best_typewell | 773 | 0.547 | 0.795 | 0.640 | 12.173 | 8.635 |
| full | provided_typewell | 773 | 0.564 | 0.795 | 0.640 | 12.317 | 8.722 |
| full | self_gr_full | 773 | 0.564 | 1.000 | 0.000 | 0.000 | 0.000 |
| full | self_gr_prefix | 773 | 0.416 | 0.956 | 0.296 | 5.946 | 3.139 |
| hidden | best_typewell | 143 | 0.592 | 0.735 | 0.728 | 11.907 | 9.012 |
| hidden | provided_typewell | 773 | 0.469 | 0.715 | 0.754 | 10.157 | 7.870 |
| hidden | self_gr_full | 773 | 0.469 | 0.995 | 0.103 | 1.358 | 0.328 |
| hidden | self_gr_prefix | 773 | 0.286 | 0.714 | 0.756 | 9.754 | 7.566 |
| prefix | best_typewell | 776 | 0.889 | 0.822 | 0.597 | 14.154 | 9.681 |
| prefix | provided_typewell | 776 | 0.916 | 0.819 | 0.602 | 14.307 | 9.803 |
| prefix | self_gr_full | 773 | 0.918 | 0.996 | 0.085 | 2.011 | 0.579 |
| prefix | self_gr_prefix | 776 | 0.918 | 1.000 | 0.000 | 0.000 | 0.000 |

## Prefix-selected typewell applied to hidden rows

- prefix だけで選んだ best typewell が hidden の provided typewell より zRMSE 改善: 0.003
- prefix だけで選んだ best typewell が hidden の provided typewell より raw RMSE 改善: 0.000
- median delta zRMSE: 0.000
- median delta raw RMSE: 0.000

## Worst provided hidden fits by grid corr

| well | provided corr | provided zRMSE | best well | best corr | best zRMSE | rank |
|---|---:|---:|---|---:|---:|---:|
| 2f19d536 | -0.332 | 1.632 | nan | - | - | - |
| d7ba4f9d | -0.331 | 1.632 | nan | - | - | - |
| d9d6d94d | -0.274 | 1.596 | nan | - | - | - |
| e46f4ef4 | -0.261 | 1.588 | nan | - | - | - |
| 347242c8 | -0.132 | 1.504 | nan | - | - | - |

## Output files

- `curve_index.csv`
- `typewell_candidate_index.csv`
- `pair_summary.csv`
- `top_typewell_candidates.csv`
- `selected_pair_exact_metrics.csv`
- `prefix_selected_hidden_metrics.csv`
- `aggregate_summary.json`
