# TVT step delta audit 2026-07-05

## Scope

- Input: `data/raw/train`
- Usable wells: 773
- Evaluation rows: 3,783,989
- Purpose: `TVT_input` known-prefix cutoff の前後で、各 step の TVT 変化量を調べる。

Definitions:

- `step_tvt_delta`: 連続行どうしの変化量、`TVT[current] - TVT[previous]`。
- `eval_step=1`: `TVT[first_eval] - last_known_TVT`。
- `relative_step=1`: cutoff 直前の 1 step と cutoff 直後の 1 step を比較する位置。
- `target_delta`: `last_known_TVT` からの累積変化量。今回の主対象ではなく参考値。

Raw outputs:

- `studies/tvt_boundary_delta_audit_20260705/eval_step_tvt_delta_summary.csv`
- `studies/tvt_boundary_delta_audit_20260705/eval_step_tvt_delta_bucket_summary.csv`
- `studies/tvt_boundary_delta_audit_20260705/boundary_relative_step_tvt_delta_summary.csv`
- `studies/tvt_boundary_delta_audit_20260705/README.md`

## Findings

### 1. 各 step の TVT 変化量はほぼ 0 centered で小さい

evaluation zone の `step_tvt_delta` は、step bucket を通して median が 0 ft、abs median が 0.01 ft で安定している。

| eval step bucket | rows | mean | p05 | p50 | p95 | p99 | abs p95 | abs p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 000_050 | 38,650 | 0.00131 | -0.04 | 0.00 | 0.0455 | 0.06 | 0.06 | 0.08 |
| 050_100 | 38,650 | 0.00137 | -0.04 | 0.00 | 0.05 | 0.07 | 0.06 | 0.08 |
| 100_250 | 115,950 | 0.00133 | -0.04 | 0.00 | 0.05 | 0.07 | 0.05 | 0.08 |
| 250_500 | 193,157 | 0.00129 | -0.04 | 0.00 | 0.04 | 0.07 | 0.05 | 0.08 |
| 500_1000 | 385,911 | 0.00085 | -0.04 | 0.00 | 0.04 | 0.06 | 0.05 | 0.07 |
| 1000_2000 | 770,087 | 0.00064 | -0.04 | 0.00 | 0.04 | 0.06 | 0.05 | 0.07 |
| 2000_plus | 2,241,584 | -0.00025 | -0.04 | 0.00 | 0.04 | 0.06 | 0.05 | 0.07 |

大半の行では 1 step あたりの TVT 変化は +/-0.05 ft 程度に収まる。まれに外れ値はあり、bucket min/max には数 ft から十数 ft のジャンプが出る。

### 2. cutoff 直前と直後の step delta 分布は近い

relative step 1 では、before mean 0.00145、after mean 0.00133、どちらも median 0、p95 0.04、p99 0.06。cutoff 直後だけ変化量が急に大きくなる傾向は見えない。

relative step 1-20 でも、after 側の p95/p99 はほぼ 0.04-0.05 / 0.06-0.07 ft。before 側も概ね同水準で、評価区間に入った直後の per-step TVT increment は prefix 末尾と同じレンジ。

### 3. 累積 TVT drift は小さい step delta の積み上がり

per-step delta は小さいが、長い tail では累積する。参考として `target_delta` の abs median は、000_050 bucket で 0.28 ft、500_1000 bucket で 6.44 ft、2000_plus bucket で 9.93 ft。

つまり PF/Beam や ML 後処理で制御すべきなのは「単発 step の大きな jump」よりも、「小さい signed delta が長く偏って積み上がる drift」。

## Implications

### PF/Beam generation

- 1 step transition の自然なレンジは、通常 `abs_step_tvt_delta` p95 0.05-0.06 ft、p99 0.07-0.08 ft 程度。
- Beam transition cost には、1 step の `|ΔTVT| > 0.08` を強く penalize する prior が使える。ただし外れ値が実在するので hard clip は危険。
- cutoff 直後を特別扱いして大きな jump を許す根拠は薄い。prefix 末尾と同じ transition prior でよい。
- long tail の累積 drift は step delta の小さい bias で起きるため、PF/Beam では transition smoothness だけでなく、GR/typewell/spatial likelihood で signed drift を誘導する必要がある。

### Current PF/Beam implementation check

- 現行採用の exp148 / exp092 surface では、PF/Beam 生成時に `ΔTVT = TVT[i] - TVT[i-1]` を直接評価する hard constraint / soft penalty は入っていない。
- Beam は typewell index の移動を `d in [-2, 2]` に制限し、`move_cost * abs(d)` を足すため、間接的な smoothness はある。ただしこれは ft 単位の `ΔTVT` p95/p99 に基づく制約ではない。
- PF 系は既知 prefix の `np.diff(TVT_input)` から初期 rate / velocity を推定し、粒子の rate/noise で滑らかに進める。生成後の per-step `ΔTVT` を clip / reject / penalize する処理ではない。
- exp146 では `dTVT/dMD + dZ/dMD` などを Beam cost に入れる audit を実装済みだが、`likpf_mean` より弱く direct Beam candidate / inference port / submit として不採用。
- exp186 では typewell late-range `candidate_pct` soft prior を PF/Beam/likelihood-PF に入れたが、これは per-step `ΔTVT` 制約ではなく、direct replacement として不採用。

### ML post-processing

- 予測列に対する causal step guard は有望。各 well の連続予測で `pred[i] - pred[i-1]` を見て、abs 0.07-0.10 ft 以上の急な jump を緩和する。
- 一方、単純な no-change や per-step zero forcing は累積 drift を潰しすぎる。step-level smoothness guard は、累積 TVT の絶対位置を決めるモデル/PF 候補とは分けて使う。
- feature としては、候補予測の `candidate_step_delta`, `candidate_abs_step_delta`, `candidate_step_delta_excess_over_train_p99`, rolling signed delta sum を verifier / selector に入れるのが妥当。

## Recommended next check

既存 OOF / submission 候補に対して、予測の `Δpred` 分布をこの train-side `ΔTVT` 分布と比較する。

1. well ごとの `pred[i] - pred[i-1]` を計算。
2. `abs(Δpred) > 0.08`、`> 0.10`、`> 0.20` の行率と well 数を見る。
3. OOF で step guard を試し、row RMSE だけでなく long-tail bucket と worst well regression を確認する。

採用候補は hard clip ではなく、外れ値だけを緩める soft guard。
