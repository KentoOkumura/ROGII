# 設計

## アプローチ

exp280が固定した各wellのunknown suffixを先頭から非重複512行blockへ分け、
各blockでexp226 `tvt_geop`へ次の13個のvertical shiftを加えた候補を維持する。

## 仮説

known prefixで得たaffine係数をcandidate blockごとに再fitせずposteriorとして固定し、
outer-train fold共通のAR(1) covarianceでGR residualをwhitenすると、
offset / scale情報を完全には捨てず、iid row-Gaussianよりtruth-nearest shiftの
MRR / top3をfold横断で改善できる。

この仮説は`affine_ar1` primaryがmatched identity-iid、保存exp280、
affine-only、AR1-onlyのすべてを事前固定marginで上回った場合だけ支持する。

```text
delta = [-80, -40, -20, -10, -5, -2, 0, 2, 5, 10, 20, 40, 80] ft
```

候補`delta`のType Well GR系列を`x_delta`、同じraw rowのhorizontal GRを`y`とする。
raw GRがfiniteな行だけを残し、元のmissing位置をまたがないcontiguous runに分ける。

### Prefix affine posterior

current-well known prefixのfinite pairだけを使い、次の観測式を置く。

```text
y = intercept + slope * x + epsilon
theta = [intercept, slope]
theta_0 = [0, 1]
V_0 = diag(20^2, 0.25^2)
sigma_w = clip(std(y_prefix - x_prefix), 10, 60)
```

`sigma_w`はexp209 / exp280と同じknown-prefix residual population stdとclipを使う。
posteriorは固定sigmaのBayesian linear regressionとして解析的に計算する。

```text
V_w = inv(inv(V_0) + X_prefix.T @ X_prefix / sigma_w^2)
m_w = V_w @ (inv(V_0) @ theta_0
             + X_prefix.T @ y_prefix / sigma_w^2)
```

finite prefix pairが64未満、Type Well GR stdが5 GR API以下、posteriorが非finite、
またはposterior mean slopeが0以下の場合、そのwellのaffine variantはineligibleとする。
identity fallbackでprimary coverageを水増ししない。

### Fold-safe AR(1) prior

各reporting foldについてouter-train wellsだけを使う。各wellのknown prefixで
`y - X m_w`を作り、元のmissing位置をまたがないlag-1 pairが64以上あるwellだけの
Yule-Walker rhoを計算する。rhoを`[-0.8, 0.8]`へclipし、
Fisher-z空間の中央値をouter-valid全wellに共通な`rho_fold`として固定する。
outer-valid well自身のrho、suffix GR、TVT truth、errorはrho推定に使わない。

block内の各contiguous finite runにstationary AR(1) whitening operator`A_rho`を適用する。

```text
u_0 = sqrt(1 - rho^2) * residual_0
u_t = residual_t - rho * residual_(t-1)
```

run間にはinnovationを作らない。

### Posterior-predictive score

blockの各runで`Z = A_rho X_delta`、`z = A_rho(y - X_delta m_w)`とし、

```text
Sigma = sigma_w^2 * I + Z @ V_w @ Z.T
log_score = -0.5 * (
    n * log(2*pi) + logdet(Sigma) + z.T @ inv(Sigma) @ z
)
```

を計算する。block scoreはrun log scoreの合計をfinite観測数で割った
mean log predictive densityとする。実装時はrank-2 Woodbury /
matrix-determinant lemmaを使ってよいが、上式との数値parityをcontract testで要求する。

### 固定2×2要因分解

全variantは同じraw-finite行、block、shift、sigma、tie orderを使う。

| variant | affine | AR(1) | 役割 |
| --- | --- | --- | --- |
| `identity_iid_matched` | `theta=[0,1]`, `V=0` | `rho=0` | matched control |
| `affine_iid` | prefix posterior | `rho=0` | affine main effect |
| `identity_ar1` | `theta=[0,1]`, `V=0` | fold-safe rho | AR main effect |
| `affine_ar1` | prefix posterior | fold-safe rho | primary |

保存exp280 raw-Gaussian scoreは別のstrong referenceとしてSHA固定で読む。
exp280はmissing補間を含むため、raw-finite matched controlと同一と主張せず、
primaryが両方を上回ることを要求する。

## 実験範囲

- 対象実験:
  `exp427_affine_ar1_whitened_gr_likelihood_readout`
- Route: `pf_beam`
- 科学的親:
  `exp280_exp226_shift_likelihood_separability_readout`
- 参考:
  `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`、
  `exp343_acf_effective_sample_likelihood_tempering_audit`、
  `exp345_exp209_time_varying_gr_affine_calibration_hmm`、
  `exp359_exp226_window_likelihood_on_exp281`、
  `exp360_typewell_reference_shift_zncc_confidence_readout`、
  `exp374_exp209_student_t_exact_hmm_emission`、
  `exp389_exp209_huber_exact_hmm_emission`
- 変更する変数:
  block GR scoreのobservation familyのみ。
- 固定する変数:
  exp226 path、13 shift、512-row non-overlap block、short-tail policy、
  Type Well interpolation / extension、known-prefix sigma clip、fold、truth-late、
  tie order、stress scope。
- inference / submission:
  無効。

## Stage 0: 0-HMM likelihood separability readout

### 対象とeligibility

- expected rows / wells / blocks: `3,783,989 / 773 / 7,787`
- block: suffix先頭から非重複512 raw rows、末尾short blockは台帳へ残す
- candidate: 13 shifts
- primary eligible block:
  finite raw GR 128行以上、block raw rowの50%以上、prefix affine eligible
- AR1 run:
  元のmissing位置をまたがない
- Type Well範囲:
  exp280と同じendpoint hold / 40 ft extension
- tie:
  config shift順

score、eligibility、fold rho、prefix posterior、input manifest、content SHAを全てfreezeした後に
だけraw horizontal TVTとhidden-like roleをjoinし、truth-nearest shiftを決める。

### 実行量

- scientific primary scores: 1 (`affine_ar1`)
- diagnostic ablation scores: 2 (`affine_iid`, `identity_ar1`)
- matched controls: 1 (`identity_iid_matched`)
- saved controls: 1 (exp280)
- reporting folds: 5
- HMM / PF / Beam runs: `0 / 0 / 0`
- model / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
- GPU / inference / submission: `0 / 0 / 0`
- parent control regeneration: 0。exp280保存scoreを読む。

### Technical AND gate

- exp226 OOF decompressed SHA、exp280 score decompressed/content SHA、
  hidden-like assignment SHAが固定値と一致する。
- row identity、well、fold、block、candidate orderが固定契約と一致する。
- eligible blockの4 matched scoreが13候補すべてfiniteである。
- eligible well fraction `>=0.90`、eligible block fraction `>=0.75`。
- affine eligible well fraction `>=0.90`。
- outer-valid wellがrho sourceに入らず、fold rhoが全fold finiteかつ
  `abs(rho)<0.8`である。
- score / eligibility / posterior / rho / manifest freeze前のsuffix truth、
  error、formation、hidden-like role readが0。
- analytic dense scoreとWoodbury scoreのsynthetic / sampled parity
  max abs error `<=1e-8`。
- stable shuffled control以外のRNGは0。shuffleもimmutable key単位のlocal RNGだけ。
- runtime `<=30,600 sec`、peak RSS `<=25 GB`。

### Scientific AND gate

primary `affine_ar1`について次を全て要求する。

- matched `identity_iid_matched`比:
  pooled MRR `>=+0.02`、top3 `>=+0.02`。
- 保存exp280 raw Gaussian比:
  pooled MRR `>=+0.01`、top3 `>=+0.01`。
- 上記2 controlに対しMRR / top3の改善foldが各`>=4/5`。
- `affine_iid`比MRR `>=+0.005`かつ改善fold`>=3/5`。
- `identity_ar1`比MRR `>=+0.005`かつ改善fold`>=3/5`。
- stable within-well/block shift-label permutationよりMRR / top3が
  `5/5 folds`で高い。
- `MD since 1000+`、hidden-like spatial、hidden-like typewell-purgedで、
  matched controlと保存exp280の双方に対してMRR / top3非悪化。
- primary top1-regret p90が保存exp280以下。

どれか1つでもFAILした場合、prior、rho、clip、support、block、shift、score family、
gateをsame-OOFで救済せずterminal closeする。partial main-effect PASSを理由に
HMM / PF実装へ進まない。

### PASS時の境界

PASSは「affine uncertainty + AR1 covarianceを持つblock scoreに追加のshift識別力がある」
ことだけを意味する。HMM / PFでのincremental factor化、overlap、state augmentation、
likelihood weight、prediction、inference、submissionは本実験の範囲外であり、
別実験番号と新しいsteeringで設計する。

## 再現性設計

- seed policy:
  real scoreはRNGなし。negative controlだけ
  `SHA256(experiment, fold, well_id, block_index)`由来local RNGを使う。
- stochastic処理:
  stable shuffled controlのみ。
- PF/Beam / likelihood-PF / seed bagging:
  なし。
- 並列処理:
  well / block / shift / run順を固定し、global RNGを使わない。
  reduction順が変わらない単位でのみ並列化する。
- runtime:
  Kaggle private CPU、GPU / internet無効。
- input SHA:
  raw well identity、exp226 OOF、exp280 score/contract、hidden-like assignment。
- feature SHA:
  prefix posterior、fold rho、eligibility、target-free 4-score table、
  saved-control alignment、negative control、manifestを記録する。
- gzip:
  raw SHAとdecompressed content SHAを分離し、後者を主証拠にする。
- model / prediction / submission SHA:
  model、prediction、submissionを生成しないため非該当。
- deterministic anchor:
  fixed-input diagnosticの数値再現性は監査するが、submission anchorではない。
- Kaggle package:
  実装・pushが別承認された場合、loose configとbootstrap config、
  Notebook body、kernel sources、input SHAを照合する。

## リスク

- likelihoodの自由度:
  candidate blockごとにaffineを再fitすると形状相関へ退化するため禁止する。
  affine posteriorはknown prefixで1回だけ固定する。
- 自己相関の過大評価:
  per-well rhoはexp343で不安定だったため、outer-train foldのFisher-z中央値だけを使う。
- 共分散の過剰適合:
  rho order、clip、shrinkage、block長を探索しない。
- missing:
  imputed GRをAR innovationへ含めず、run境界を保持する。exp280 saved controlとは
  support差を明示し、matched controlを別に置く。
- 相関指標への退化:
  affine posterior uncertainty、sigma、log determinantを含むproper predictive densityを
  使い、Pearson / ZNCCをscoreへ含めない。
- 複合変更:
  fixed 2×2 ablationでaffine / AR1のmain effectを分離し、primaryが両単独variantを
  上回らない場合は不採用にする。
- CV/LB:
  Stage 0はshift-rank diagnosticで予測CV / LBではない。
- runtime / memory:
  dense 512×512 inverseをproduction実装せず、rank-2式を使う。
  dense式は小型parity testだけに限定する。
- negative transfer:
  exp345 / exp374 / exp389は平均改善とworst-well悪化を示したため、
  stress scopeとtop1-regret p90をAND gateに含める。
- closed-family rescue:
  exp343 / exp359 / exp360のFAILを再分類せず、それぞれのtemperature、
  heuristic window score、ZNCCを再利用しない。
