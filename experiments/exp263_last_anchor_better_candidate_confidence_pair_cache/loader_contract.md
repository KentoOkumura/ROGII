# exp263 candidate cache loader contract

## 目的

`candidate_values` に保存するのは core 12 primitive だけです。8 pair、w500 alias、3 named
combination は `CandidateCache.materialize()` が要求された fold / row slice だけを読み、primitive
から決定的に再構成します。pair / triple の全行tensorや再帰closureは保存しません。

## 配置

```text
<cache-root>/
  candidate_catalog.json
  candidate_values/<candidate_id>/fold=<0..4>/part-000.parquet
  candidate_confidence/<candidate_id>/fold=<0..4>/part-000.parquet
  outer_fold_eligibility.csv
  pair_shortlist.csv
  pair_readout.csv
  named_combinations.json
  cache_manifest.json
  small_parity_sample.parquet
```

primitive partitionのkeyは `id`, `well`, `well_row_idx`, `outer_fold`, `md_since` です。
canonical exp072の行順、ID、well-row spanと完全一致させます。`candidate_tvt` と派生数値は
float32です。

## API

```python
from pathlib import Path

from candidate_cache_loader import CandidateCache

cache = CandidateCache(Path("/kaggle/input/exp263-last-anchor-cache"))

# primitiveをfold/chunk単位で読む
exp226 = cache.load_primitive(
    "exp226_k16",
    fold=0,
    row_slice=slice(0, 100_000),
)

# fixed pairをprimitiveから再構成する
pair = cache.materialize(
    "exp226_k16__exact_hmm",
    fold=0,
    row_slice=slice(0, 100_000),
    include_confidence=True,
)

# w500 aliasと固定named formula
w500 = cache.materialize(
    "blend_likpf_hmm_w500",
    fold=0,
    row_slice=slice(0, 100_000),
)
fixed = cache.materialize(
    "exp226_w500_50_50",
    fold=0,
    row_slice=slice(0, 100_000),
)

# outer-crossfit式は対象outer foldを必須にする
diagnostic = cache.materialize(
    "exp226_selfgr_a070_likpf_outer_convex",
    fold=0,
    row_slice=slice(0, 100_000),
)
```

## selectable guard

primitive selectorでは `blend_likpf_hmm_w500` と、その親 `likpf_mean` / `exact_hmm` を同時に
selectableにしません。

```python
cache.validate_selectable(["exp226_k16", "likpf_mean", "exact_hmm"])

# ValueError: w500とprimitive親を同時登録しない
cache.validate_selectable(["blend_likpf_hmm_w500", "exact_hmm"])
```

pairとnamed combinationを同時に再帰展開するclosureも禁止します。新しい組合せを追加するときは、
個別のnamed formula、deployability tier、保存OOF根拠を実験契約へ追加します。

## confidence

`candidate_confidence` はsource artifactに実在するtarget-free列だけを保存します。sourceにない
診断値はNaN、`confidence_valid=False`、`confidence_missing_fields`へ記録し、0や推測値を作りません。
HMM sourceのwell-level total log-likelihoodはcanonical well row countで割った
`loglik_per_row`だけを決定的派生し、元の`source_loglik`も残します。
異なるfamily間の `sigma_tvt` は同じ尺度と仮定せず、parent別のnamespaced列のまま返します。同familyで
同じ意味のcommon slotだけ、min/max/mean/range ratioを生成できます。

## target-derived readoutの隔離

`candidate_catalog.json` のRMSE、`pair_shortlist.csv`、`pair_readout.csv`、
`outer_fold_eligibility.csv` は監査・選択根拠です。これらのtarget-derived列をrow featureとして
`candidate_values` / `candidate_confidence`へjoinしてはいけません。outer eligibilityは各outer-valid
foldを除いた4 foldsだけで計算します。

## Stage 1

current-test parityで許可するのはraw-test-ready 6 primitive、raw-test 5 pair、固定
`exp226_w500_50_50`だけです。train-only 6 primitive、outer-convex diagnostic、HMM+LGB、selector / TVT
model outputsは暗黙に昇格させません。Stage 1はraw competition testから6 primitiveを再生成し、
fixed formulaの`submission.csv`と後段selector用`current_test_formula_parity.parquet`を作ります。

同Parquetには候補値12列に加え、次の21列を
`confidence__<primitive_id>__<field>`で保存します。

- exp226 K16: `confidence_valid`, `geometry_gr_delta`
- self-GR HMM: `confidence_valid`, `sigma_tvt`, `source_loglik`, `loglik_per_row`,
  `candidate_finite_source`, `selfgr_quality`, `selfgr_peak_tvt`, `score_margin`,
  `selfgr_typewell_agreement`, `selfgr_valid`
- likPF: `confidence_valid=False`のみ。sourceにないscalarを推測しません。
- exact HMM: `confidence_valid`, `sigma_tvt`, `source_loglik`, `loglik_per_row`
- PF-ANCC: `confidence_valid`, `sigma_tvt`
- Beam mean: `confidence_valid`, `beam_family_std`

HMMの`source_loglik`はwell-level totalを各rowへ保持し、`loglik_per_row`はそのwellのunknown row数で
割ります。formula候補のconfidenceは平均せず、後段が親primitiveのnamespaceから展開します。required列の
欠損・非finite・invalidはfail-closedです。
