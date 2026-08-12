# exp221 lgb_oof_gaussian_emission_hmm_on_exp148 design

## Approach

Use exp209 exact HMM as the base smoother. Keep the HMM grid, transition, GR emission, and runtime settings fixed. For each evaluation row and each HMM state TVT, add:

```text
logP_lgb = -0.5 * ((state_tvt - pred_tvt_lgb) / sigma)^2
```

The combined emission is `GR_loglik + lambda_lgb * logP_lgb`.

## Initial Runtime Plan

- `feature_cache.hmm.outer_workers=2`
- `runtime.numba_num_threads=2`
- Initial variants: 3
- LightGBM boosters: 0
- GPU: disabled

The deferred 3x3 grid is intentionally not active in the first run because each HMM variant is a full DP pass.

## Data Flow

1. Read raw train horizontal/typewell files.
2. Read exp148 OOF `lgb_mean` prediction as fixed emission center.
3. Run HMM once per active sigma/lambda variant.
4. Write a wide train feature cache with one prediction/std/loglik block per variant.
5. Read saved exp072 feature cache for baseline candidates.
6. Read exp148/exp193 OOF predictions for ML baselines.
7. Produce overall, bucket, exp115 hidden-like, by-well, std calibration, and step-delta readouts.

## Leakage Guard

- Unknown suffix `TVT` is metric-only.
- LGB OOF predictions are fixed inputs.
- Sigma/lambda are global grid values from config.
- No row-wise adjustment from true errors.
- No inference or submission until train-side gates pass.

## Risks

- Runtime scales linearly with active HMM variants.
- Small sigma can make HMM over-adhere to LGB and amplify ML errors.
- Large sigma can make the LGB emission ineffective.
- Hidden-like and worst-well regressions may be more important than small global RMSE gains.
