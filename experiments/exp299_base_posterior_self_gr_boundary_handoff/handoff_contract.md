# exp299 base-posterior self-GR boundary handoff contract

## 目的

range外candidateではself-GRを一切参照せず、境界通過時にはrange内stateだけが正boostを持つ相対障壁を作らない。exp223 raw motif signalをsupport内条件付き順位付けへ限定する。

## 固定数式

```text
L = min(finite visible-prefix TVT_input)
U = max(finite visible-prefix TVT_input)
S = {s | L <= grid[s] <= U}

p0_t(s) = Type-Well-only exact-HMM posterior
mu0_t   = sum_s p0_t(s) * grid[s]
m0_t    = sum_{s in S} p0_t(s)
d0_t    = max(0, min(mu0_t - L, U - mu0_t))
g_t     = m0_t * clip(d0_t / 12.0, 0, 1)

b_t(s) = clip(exp223_centered_self_gr_surface_t(s), 0, 1)
r_t(s) = 0.07 * exp223_quality_t * b_t(s)

if m0_t <= eps or g_t == 0:
    C_t(s) = 0 for all s
else:
    z_t = log(sum_{s in S} p0_t(s) * exp(g_t * r_t(s)) / m0_t)
    C_t(s) = g_t * r_t(s) - z_t  if s in S else 0

emission_t(s) = typewell_emission_t(s) + C_t(s)
```

## 不変条件

- Pass Aはself-GRなし、Pass BはPass A freezeだけをcontrollerに使う。
- `C_t(s outside S) == 0.0`。
- `mu0_t <= L`または`mu0_t >= U`なら`C_t(all states) == 0.0`。
- `0 <= g_t <= 1`。
- `sum_S p0_t(s) * exp(C_t(s)) == sum_S p0_t(s)`をrelative tolerance `1e-6`以内で満たす。
- exp223 raw surface / qualityはhandoff前にexact parity。
- final prediction、Pass B posterior、true TVT、error、oracle、exp223/296 predictionをcontrollerへ使わない。
- final predictionをknown rangeへclipしない。
- finite known TVTまたはbase inside massがなければself-GR all-neutral。

## Variantとcount

- variant: `hmm_selfgr_base_posterior_conditional_handoff_a070_c100`
- scientific variants: 1
- Pass A / Pass B HMM well-runs: `773 / 773`
- total HMM well-runs: `1,546`
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- parent control retraining / GPU / inference / submission: `0 / 0 / 0 / 0`

## 停止条件

technicalとperformanceの全hard gateをPASSした場合だけscientific supportとする。1条件でもFAILならhandoff式、12-ft fade、conditional normalizer、support、alpha、clip、window、top-k、thresholdを変更する救済variantを作らず閉じる。

PASSでもexp209 HMM/likPF blend 10.269696以下、raw-test-safe two-pass設計、ユーザー別承認なしにinferenceへ進めない。

## 現在

正規train Notebookを採用し、Kaggle private CPU version 2を完了した。exp209 parityはmax/mean abs `0/0 ft`でPASSしたが、technicalはrow gate maxの`2.9e-15`超過により24/25、performanceはcandidate RMSE `11.789577561`、exp223比`+0.439634615 ft`、改善0/5 foldsで2/11 PASS。停止条件どおりnegative resultとしてbranchを閉じ、数式救済、version 3、inference、submissionへ進めない。
