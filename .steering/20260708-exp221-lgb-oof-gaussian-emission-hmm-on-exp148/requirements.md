# exp221 lgb_oof_gaussian_emission_hmm_on_exp148 requirements

## Goal

Implement the `lgb_oof_gaussian_emission_hmm_on_exp148` backlog as a train-side no-training audit.

## Scope

- Create `exp221_lgb_oof_gaussian_emission_hmm_on_exp148`.
- Route is `ensemble`.
- Parent references are exp209 exact HMM, exp148 OOF prediction, exp193 OOF prediction, and exp115 hidden-like split.
- Add LightGBM OOF point predictions as Gaussian emission terms inside the exp209 HMM.
- Initial active grid is exp148 `lgb_mean` with `sigma=[8, 12, 20]` and `lambda=0.50`.
- Compare against exp148 `lgb_mean`, exp193 `lgb_mean`, and exp072 `likpf_mean`.
- Save overall, distance bucket, hidden-like subgroup, by-well, HMM std calibration, and step-delta readouts.

## Non-Goals

- Do not train LightGBM.
- Do not regenerate exp072 by default.
- Do not create raw-test inference or `submission.csv`.
- Do not row-wise tune sigma/lambda from true TVT, OOF absolute error, oracle best, or true-error rank.
- Do not use `pred_tvt_lgb` as hard replacement or posthoc direct correction.

## Success Criteria

- Static validation passes.
- Kaggle train can run CPU-only with 0 boosters and no parent/control retraining.
- Generated HMM+LGB candidates align 1:1 with exp072 baseline and LGB OOF predictions.
- Readout identifies whether HMM+LGB improves or regresses versus exp148 / exp193 without hiding bucket or worst-well regressions.
