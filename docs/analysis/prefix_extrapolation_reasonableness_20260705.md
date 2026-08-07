# Prefix Extrapolation Reasonableness Audit 2026-07-05
## Scope
- Input: `data/raw/train` train horizontal wells, 773 usable wells.
- Fits use only `TVT_input` known-prefix rows. Metrics use the hidden/evaluation rows where train `TVT` is available.
- This is a pre-experiment diagnostic, not a submission candidate.

## Findings

- 直接外挿候補としては `anchor` が最良。`TVT ~ MD`、`TVT ~ Z`、`TVT ~ MD+Z`、`U=TVT+Z` の直接外挿はいずれも global RMSE で `anchor` を大きく下回った。
- 例外的に、`U=TVT+Z` を known prefix tail 100 rows から外挿した候補を 10% だけ anchor に混ぜる `blend10_u_md_tail100_anchor` は RMSE 15.909853 -> 15.797007 と -0.112846 改善した。ただし MAE は 11.196480 -> 11.203994 と微悪化し、max well regression は +17.384692 RMSE ある。
- `U=TVT+Z` tail 外挿は near-prefix では強い。`u_md_tail100_anchor` は eval step `000_050` で RMSE 0.351898、`050_100` で 0.924661、`100_250` で 2.263470、`250_500` で 4.953467、`500_1000` で 10.370862 と anchor より良い。一方 `1000_plus` では 51.632173 まで壊れ、anchor 17.253401 に大きく負ける。
- true drift bucket では、`anchor` は drift 20 ft 未満で強い。外挿 blend は drift 40 ft 以上では助かるが、true drift は test では直接見えない。prefix-only proxy の Spearman 相関は弱く、guard は簡単ではない。
- `TVT ~ MD` の全 prefix slope は破綻しており、`md_all_anchor` は RMSE 1260.165362。tail 50/200 slope や `MD+Z` slope も well 単位の最大 regression が 1000 ft 級になるため、PF/Beam generation に直接 hard prior として入れるべきではない。

## Decision

- PF/Beam 生成で試すなら、候補は `U=TVT+Z` の tail slope 系だけに絞る。
- 外挿 path を直接候補・hard prior・hard prune にしない。
- 使う場合は `alpha <= 0.10`、near-prefix / distance-aware fade、または selector/confidence feature として扱う。`1000_plus` longtail では原則 fade out する。
- 次に実験化するなら「PF/Beam candidate scoring / selector に prefix extrapolation confidence を入れる」方が、「PF/Beam transition を外挿で縛る」より安全。

## Top Overall Metrics

| rows | rmse | mae | bias | within5 | within10 | abs_p50 | abs_p90 | abs_p95 | abs_p99 | method | delta_rmse_vs_anchor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3783989 | 15.797007 | 11.203994 | -1.409933 | 0.350965 | 0.582764 | 7.994080 | 24.821497 | 32.785133 | 51.260445 | blend10_u_md_tail100_anchor | -0.112846 |
| 3783989 | 15.904729 | 11.273985 | -1.462957 | 0.349971 | 0.577786 | 8.035000 | 24.916000 | 32.759998 | 51.301998 | blend10_u_md_tail30_median_anchor | -0.005124 |
| 3783989 | 15.909853 | 11.196480 | -1.595987 | 0.349544 | 0.578629 | 8.030000 | 24.680000 | 32.340000 | 53.130001 | anchor | 0.000000 |
| 3783989 | 18.001900 | 12.549743 | -1.130853 | 0.338467 | 0.557868 | 8.444964 | 29.124601 | 38.474335 | 58.549473 | blend25_u_md_tail100_anchor | 2.092047 |
| 3783989 | 18.502983 | 12.417983 | -0.990613 | 0.334650 | 0.552463 | 8.617669 | 27.309835 | 35.927746 | 63.170105 | blend10_md_tail200_anchor | 2.593130 |
| 3783989 | 18.555390 | 12.753112 | -1.263411 | 0.340082 | 0.550346 | 8.567500 | 29.469999 | 38.932499 | 61.119999 | blend25_u_md_tail30_median_anchor | 2.645537 |
| 3783989 | 20.528005 | 13.167433 | -0.999976 | 0.338615 | 0.551680 | 8.538926 | 29.003809 | 40.851669 | 75.864105 | blend10_mdz_tail200_anchor | 4.618152 |
| 3783989 | 32.660594 | 17.861749 | -0.082551 | 0.282986 | 0.475143 | 10.807159 | 40.751434 | 55.634018 | 92.295876 | blend25_md_tail200_anchor | 16.750742 |
| 3783989 | 37.509081 | 18.807133 | -0.105959 | 0.306957 | 0.503912 | 9.866768 | 40.413235 | 65.116325 | 153.799500 | blend25_mdz_tail200_anchor | 21.599228 |
| 3783989 | 46.202841 | 28.624309 | 0.264551 | 0.257324 | 0.392981 | 15.314941 | 73.565956 | 101.149879 | 172.665344 | u_md_tail100_anchor | 30.292988 |
| 3783989 | 49.493155 | 29.667997 | -0.265682 | 0.258457 | 0.394557 | 15.180000 | 76.690002 | 106.800003 | 177.660004 | u_md_tail30_median_anchor | 33.583302 |
| 3783989 | 63.544332 | 38.713713 | -5.835380 | 0.190276 | 0.307379 | 21.916161 | 93.710793 | 127.346992 | 228.882874 | mdz_all_anchor | 47.634479 |

## Direct Extrapolation Methods

| rows | rmse | mae | bias | within5 | within10 | abs_p50 | abs_p90 | abs_p95 | abs_p99 | method | delta_rmse_vs_anchor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3783989 | 15.909853 | 11.196480 | -1.595987 | 0.349544 | 0.578629 | 8.030000 | 24.680000 | 32.340000 | 53.130001 | anchor | 0.000000 |
| 3783989 | 46.202841 | 28.624309 | 0.264551 | 0.257324 | 0.392981 | 15.314941 | 73.565956 | 101.149879 | 172.665344 | u_md_tail100_anchor | 30.292988 |
| 3783989 | 49.493155 | 29.667997 | -0.265682 | 0.258457 | 0.394557 | 15.180000 | 76.690002 | 106.800003 | 177.660004 | u_md_tail30_median_anchor | 33.583302 |
| 3783989 | 63.544332 | 38.713713 | -5.835380 | 0.190276 | 0.307379 | 21.916161 | 93.710793 | 127.346992 | 228.882874 | mdz_all_anchor | 47.634479 |
| 3783989 | 66.215766 | 43.168559 | -1.287362 | 0.158112 | 0.272938 | 25.610292 | 105.012863 | 143.067459 | 239.489258 | u_md_all_anchor | 50.305913 |
| 3783989 | 107.494824 | 86.847114 | -14.613285 | 0.040729 | 0.076997 | 77.019997 | 173.110001 | 205.500000 | 267.489990 | u_const_anchor | 91.584971 |
| 3783989 | 109.052922 | 87.778942 | -20.065813 | 0.041047 | 0.077791 | 77.242409 | 176.537018 | 209.058533 | 272.344513 | z_all_anchor | 93.143069 |
| 3783989 | 113.634734 | 56.441488 | 3.962154 | 0.168673 | 0.272855 | 27.001177 | 139.101135 | 194.966614 | 351.037689 | md_tail50_anchor | 97.724881 |
| 3783989 | 123.763726 | 55.314442 | 4.457757 | 0.162741 | 0.269859 | 26.789537 | 135.948242 | 185.841782 | 324.899078 | md_tail200_anchor | 107.853873 |
| 3783989 | 139.752390 | 55.753488 | 4.364124 | 0.226832 | 0.353325 | 18.925007 | 121.347305 | 232.849762 | 589.638733 | mdz_tail200_anchor | 123.842537 |
| 3783989 | 146.402516 | 63.005331 | 9.272112 | 0.170122 | 0.281251 | 25.154814 | 133.609375 | 242.027573 | 655.893555 | z_tail200_anchor | 130.492663 |
| 3783989 | 1260.165362 | 1030.629375 | 1030.629375 | 0.002722 | 0.005636 | 921.791443 | 2016.922119 | 2346.777344 | 3145.211670 | md_all_anchor | 1244.255509 |

## Step Bucket Focus

| rows | rmse | mae | bias | within5 | within10 | abs_p50 | abs_p90 | abs_p95 | abs_p99 | method | bucket_type | bucket |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 37877 | 0.295526 | 0.132232 | 0.000584 | 1.000000 | 1.000000 | nan | nan | nan | nan | u_md_tail30_median_anchor | eval_step | 000_050 |
| 37877 | 0.351898 | 0.153788 | 0.001936 | 1.000000 | 1.000000 | nan | nan | nan | nan | u_md_tail100_anchor | eval_step | 000_050 |
| 37877 | 0.573912 | 0.244991 | -0.002526 | 0.998522 | 1.000000 | nan | nan | nan | nan | mdz_tail200_anchor | eval_step | 000_050 |
| 37877 | 0.653473 | 0.378123 | 0.013120 | 0.998812 | 1.000000 | nan | nan | nan | nan | md_tail200_anchor | eval_step | 000_050 |
| 37877 | 0.942078 | 0.501659 | -0.036233 | 0.997545 | 0.999076 | nan | nan | nan | nan | anchor | eval_step | 000_050 |
| 38650 | 0.843075 | 0.470354 | 0.004903 | 0.997050 | 1.000000 | nan | nan | nan | nan | u_md_tail30_median_anchor | eval_step | 050_100 |
| 38650 | 0.924661 | 0.512219 | 0.008933 | 0.998836 | 1.000000 | nan | nan | nan | nan | u_md_tail100_anchor | eval_step | 050_100 |
| 38650 | 1.859929 | 0.879149 | 0.005638 | 0.974463 | 0.993092 | nan | nan | nan | nan | mdz_tail200_anchor | eval_step | 050_100 |
| 38650 | 1.891774 | 1.230188 | 0.045910 | 0.982794 | 0.996843 | nan | nan | nan | nan | md_tail200_anchor | eval_step | 050_100 |
| 38650 | 2.333592 | 1.453057 | -0.101161 | 0.976093 | 0.996171 | nan | nan | nan | nan | anchor | eval_step | 050_100 |
| 3012442 | 17.253401 | 12.519070 | -1.849619 | 0.292423 | 0.523193 | nan | nan | nan | nan | anchor | eval_step | 1000_plus |
| 3012442 | 51.632173 | 34.778107 | 0.380688 | 0.143916 | 0.272616 | nan | nan | nan | nan | u_md_tail100_anchor | eval_step | 1000_plus |
| 3012442 | 55.317362 | 36.074624 | -0.277809 | 0.143392 | 0.271905 | nan | nan | nan | nan | u_md_tail30_median_anchor | eval_step | 1000_plus |
| 3012442 | 138.317416 | 66.894779 | 5.502481 | 0.082865 | 0.166154 | nan | nan | nan | nan | md_tail200_anchor | eval_step | 1000_plus |
| 3012442 | 156.268336 | 67.788197 | 5.325611 | 0.127488 | 0.244346 | nan | nan | nan | nan | mdz_tail200_anchor | eval_step | 1000_plus |
| 115950 | 2.200289 | 1.319403 | 0.014274 | 0.952954 | 0.994023 | nan | nan | nan | nan | u_md_tail30_median_anchor | eval_step | 100_250 |
| 115950 | 2.263470 | 1.347013 | 0.023715 | 0.944985 | 0.996645 | nan | nan | nan | nan | u_md_tail100_anchor | eval_step | 100_250 |
| 115950 | 4.617348 | 3.041094 | -0.214309 | 0.804580 | 0.969099 | nan | nan | nan | nan | anchor | eval_step | 100_250 |
| 115950 | 5.204566 | 2.397350 | 0.148823 | 0.879043 | 0.957163 | nan | nan | nan | nan | mdz_tail200_anchor | eval_step | 100_250 |
| 115950 | 5.290004 | 3.159958 | 0.130173 | 0.815179 | 0.959577 | nan | nan | nan | nan | md_tail200_anchor | eval_step | 100_250 |
| 193158 | 4.953467 | 3.225654 | -0.135925 | 0.783245 | 0.938289 | nan | nan | nan | nan | u_md_tail100_anchor | eval_step | 250_500 |
| 193158 | 5.112692 | 3.272273 | -0.157004 | 0.807168 | 0.937973 | nan | nan | nan | nan | u_md_tail30_median_anchor | eval_step | 250_500 |
| 193158 | 7.646472 | 5.434162 | -0.525112 | 0.576223 | 0.835176 | nan | nan | nan | nan | anchor | eval_step | 250_500 |
| 193158 | 12.657560 | 6.054561 | 0.549252 | 0.671445 | 0.843444 | nan | nan | nan | nan | mdz_tail200_anchor | eval_step | 250_500 |
| 193158 | 12.858885 | 7.383630 | 0.214715 | 0.493415 | 0.768014 | nan | nan | nan | nan | md_tail200_anchor | eval_step | 250_500 |
| 385912 | 10.370862 | 7.105695 | -0.317832 | 0.525591 | 0.757867 | nan | nan | nan | nan | u_md_tail100_anchor | eval_step | 500_1000 |
| 385912 | 10.847476 | 7.209993 | -0.362762 | 0.526592 | 0.779818 | nan | nan | nan | nan | u_md_tail30_median_anchor | eval_step | 500_1000 |
| 385912 | 11.043234 | 8.232334 | -0.870065 | 0.418901 | 0.682549 | nan | nan | nan | nan | anchor | eval_step | 500_1000 |
| 385912 | 27.535596 | 15.387252 | 0.604697 | 0.260533 | 0.478342 | nan | nan | nan | nan | md_tail200_anchor | eval_step | 500_1000 |
| 385912 | 28.187780 | 13.660769 | 0.899771 | 0.433192 | 0.649731 | nan | nan | nan | nan | mdz_tail200_anchor | eval_step | 500_1000 |

## Worst Direct Well-Level Regressions

| method | mean_well_rmse | median_well_rmse | p90_well_rmse | max_well_rmse | improved_wells_vs_anchor | worse_wells_vs_anchor | max_well_regression_vs_anchor | best_well_improvement_vs_anchor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| md_all_anchor | 1111.837142 | 1086.350989 | 1593.826737 | 2979.325654 | 0 | 773 | 2932.286405 | 82.330944 |
| z_tail200_anchor | 69.430681 | 33.893835 | 138.560513 | 1685.688455 | 103 | 670 | 1660.326594 | -50.996416 |
| mdz_tail200_anchor | 59.754727 | 27.345660 | 134.966210 | 1541.511611 | 166 | 607 | 1516.532840 | -41.656592 |
| md_tail200_anchor | 59.329557 | 35.075394 | 131.203897 | 1534.530951 | 102 | 671 | 1487.491703 | -39.504793 |
| md_tail50_anchor | 60.682858 | 37.987961 | 132.368799 | 1392.426085 | 99 | 674 | 1327.067426 | -54.038748 |
| mdz_all_anchor | 43.030622 | 30.999187 | 87.010174 | 388.946188 | 136 | 637 | 353.340510 | -45.358232 |
| u_md_tail30_median_anchor | 33.099195 | 22.442047 | 76.538676 | 358.099517 | 195 | 578 | 346.887612 | -50.408379 |
| u_md_all_anchor | 47.879082 | 36.191028 | 101.019518 | 282.351107 | 125 | 648 | 257.336373 | -42.799283 |

## Files

- `studies/prefix_extrapolation_reasonableness_20260705/overall_metrics.csv`
- `studies/prefix_extrapolation_reasonableness_20260705/well_summary.csv`
- `studies/prefix_extrapolation_reasonableness_20260705/by_well_metrics.csv`
- `studies/prefix_extrapolation_reasonableness_20260705/step_bucket_metrics.csv`
- `studies/prefix_extrapolation_reasonableness_20260705/drift_bucket_metrics.csv`
