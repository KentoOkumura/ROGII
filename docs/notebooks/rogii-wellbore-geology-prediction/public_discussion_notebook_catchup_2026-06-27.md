# ROGII public discussion / notebook catch-up 2026-06-27

調査日: 2026-06-27

## Source

- Competition: `rogii-wellbore-geology-prediction`
- Discussion listing: `kaggle competitions topics list --sort-by new/recent`
- Notebook listing: `kaggle kernels list --sort-by dateRun/scoreAscending --page-size 80 -v`
- Pulled notebooks:
  - `docs/notebooks/rogii-wellbore-geology-prediction/date_run_recent_20260627/`
  - `docs/notebooks/rogii-wellbore-geology-prediction/score_ascending_20260627/`

## Executive Summary

1. Public notebook ecosystem has shifted from standalone PF / Ridge ideas to a dense fork cluster around `dual-pipeline`, `koolbox`, `fle3n`, `ravaghi` artifacts, and Gold/visible-prefix overlays.
2. The apparent public frontier is now dominated by notebooks titled around LB `7.159`, `7.201`, and `7.295`. These are mostly public-LB measured titles and forks; treat them as replay/audit targets, not validated private-safe methods.
3. Discussion consensus is warning against reforking/seed-weight tuning without honest CV. New technical debate focuses on whether remaining error is learnable from legal data, bimodal +/-15 ft datum ambiguity, and GR/typewell degeneracy.
4. Official update: private test outlier excluded from scoring on 2026-06-11; Working Note Awards added, deadline 2026-07-06 11:59 UTC.
5. For our repo, the immediate useful direction is not another blind public fork. Audit the public dual-pipeline family against our hidden-safe replay rules, then separately test one focused idea: candidate ranking / ambiguity-aware midpoint hedge on our OOF.

## Latest Discussions

| Topic | Date | Signal |
| --- | --- | --- |
| [New: Working Note Awards! Submit by July 6](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/709495) | 2026-06-18 | Two optional $2,500 awards for medal-zone teams; working note deadline is 2026-07-06 11:59 UTC. |
| [Private Test Update and Rescore](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/707695) | 2026-06-11 | One private-test outlier well excluded from scoring; public LB unchanged. |
| [What wrong with TVT](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/714514) | 2026-06-26 | Participants still hitting 8-13 LB ask what top solutions are doing differently; replies point to generalization, regularization, and likely public overfit. |
| [Clarification on X,Y coordinate system...](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/713987) | 2026-06-25 | No host confirmation yet. Treat coordinates as local/relative; georeferencing may be leakage-prone or impossible. |
| [Stop reforking - where the error actually lives](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/712037) | 2026-06-22 | Core warning: public `~7.2` cluster may be refork/seed variance. GR has coarse signal but may not identify fine per-well slope under honest grouping. |
| [The +/-15 ft datum: why some wells are unsolvable](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/711878) | 2026-06-22 | Hypothesis: cyclic Eagle Ford GR signatures create two plausible datum modes separated by about 15-30 ft; midpoint can be RMSE-optimal when ambiguous. |
| [Problem Breakdown](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/708367) | 2026-06-15 | Good conceptual explanation of TVT/geosteering; late comments discuss GR sensor rotation / denoising. |
| [Formation Columns Are Derived from Typewell...](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/708167) | 2026-06-14 | Formation columns are effectively one base surface plus constant offsets, with anomalous ANCC cases. Useful for reducing feature assumptions. |
| [Are Public Notebooks Overfitting to the LB?](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/707915) | 2026-06-12 | Late comments quantify public score band from reruns/particle reseeding; reinforces replay with fixed seeds and OOF-based gates. |

## Public Notebook Landscape

### Score-ascending frontier

| Ref | Listed title / signal | Local path | Notes |
| --- | --- | --- | --- |
| `degnonguidi/public-score-rogii-lb-7-159` | `Public Score ROGII [LB: 7.159]` | `score_ascending_20260627/degnonguidi__public-score-rogii-lb-7-159/` | Rebuilt dual-pipeline, explicitly documents Gold overlay as public-LB-only and off by default in the visible code. |
| `aevionlabs/public-score-rogii-lb-7-159` | `Public Score ROGII [LB: 7.159]` | `score_ascending_20260627/aevionlabs__public-score-rogii-lb-7-159/` | Same family; external datasets include `koolbox`, `fleongg`, `ravaghi`, and extra artifacts. |
| `bernubritz/rogii-lb7295-public-rebuild` | `ROGII LB7295 Public Rebuild` | `score_ascending_20260627/bernubritz__rogii-lb7295-public-rebuild/` | Strong fork baseline. Uses GPU, offline `koolbox`, `fleongg/rogii-claude-models-pub`, and `ravaghi` artifacts. |
| `boristown/rogii-lb7295-public-rebuild-v1-xr-recovery` | `LB7295 ... XR Recovery` | `score_ascending_20260627/boristown__rogii-lb7295-public-rebuild-v1-xr-recovery/` | Latest 2026-06-27 active fork; enables `ROGII_GOLD_PREFIX_CAL=1`, conservative profile, guarded revert comments. |
| `fleongg/rogii-wellbore-geology-dual-pipeline-lb-7-2` | `dual-pipeline (LB ~7.2)` | `score_ascending_20260627/fleongg__rogii-wellbore-geology-dual-pipeline-lb-7-2/` | Readable dual-pipeline architecture: two branches, LightGBM/CatBoost/Ridge meta, PF/beam/physics features, blend 0.55/0.45, guarded override. |
| `pilkwang/rogii-bimodal-hedge-geosteering-rebuild` | `Bimodal-Hedge Geosteering Rebuild` | `score_ascending_20260627/pilkwang__rogii-bimodal-hedge-geosteering-rebuild/` | Contact/geosteering route with profile presets, ridge/PF selector blend, projection blend, bimodal detector. |
| `hongweiluan/rogii-wellbore-v6-e2e` | Recent CPU script, 2026-06-27 | `date_run_recent_20260627/hongweiluan__rogii-wellbore-v6-e2e/` | More ML-heavy: 5 LGBM + XGBoost + CatBoost + optional CNN/DWT/formation blend artifacts. Worth reading as a different route, not the current public fork cluster. |

### Current clusters

- `dual-pipeline / public rebuild`: strongest and most copied. It combines physics/PF style trajectories with GBM/Ridge stacks and guarded overrides. Risk: many variants are public-LB tuned overlays and artifact-dependent.
- `Gold visible-prefix overlay`: used in many `7.159/7.201/7.295` titles. It exploits visible prefix / train-test overlap style checks. Treat as a separate audit flag; it may be public-LB boosting and not private-safe.
- `target-free / contact / bimodal hedge`: Pilkwang and souldrive threads point toward ambiguity-aware trajectory selection or midpoint hedging. This is more conceptually useful for our own OOF work than direct fork copying.
- `ML artifact route`: Hongwei/Leemarc/Nickson style notebooks load external trained models and run formation KNN / DenseANCC / CNN or GBM ensembles. Useful to inspect, but high artifact dependency.
- `direct overlap lookup diagnostics`: Some latest notebooks explicitly lookup train MD->TVT overlap for visible/example test wells. This should be treated as diagnostic or public-only unless hidden-test behavior is proven valid inside the code-competition environment.

## Risks / Interpretation

- Public score titles are not an independent source of truth. They are notebook titles or markdown claims.
- Most strong notebooks require external datasets/artifacts and GPU metadata. Any replay should first inventory dataset sources and check offline compatibility.
- The public notebooks increasingly encode public-board feedback loops: `GOLD_PROFILE`, `ROGII_GOLD_PREFIX_CAL`, blend weights, conservative/balanced profiles, and seed sweeps.
- Discussion now strongly suggests public LB variance from reseeding/reforking. Our experiment decisions should use OOF, by-well diagnostics, and hidden-safe replay checks before adopting any public fork.

## Recommended Next Actions

1. Audit `degnonguidi/public-score-rogii-lb-7-159` and `fleongg/rogii-wellbore-geology-dual-pipeline-lb-7-2` as read-only references: list external artifacts, identify Gold/public-only branches, and map which components are hidden-safe.
2. Compare their PF/beam/GBM candidate generation against our exp073/exp086 readouts; focus on candidate pool quality and selector/ranker features, not final public blend weights.
3. Add an OOF test for bimodal ambiguity: per-well residual distribution, alternative datum candidates around +/-15-30 ft, and whether midpoint hedging reduces worst-well RMSE.
4. Do not push a Kaggle run that retrains control/public baselines until variant count, fold count, booster count, and parent reuse are documented in `SESSION_NOTES.md`.
5. If chasing a submission, choose one replay target and run it unchanged first. Record runtime, metadata, output SHA, `submission.csv` validation, and LB before modifying.

## Local Changes Made During Catch-up

- Updated `.agents/skills/kaggle-notebook-fetch/scripts/fetch_top_notebooks.py` to ignore Kaggle CLI version-warning lines in stdout before CSV parsing.
- Saved fresh notebook listings and pulled 25 `dateRun` plus 25 `scoreAscending` notebooks under the two 2026-06-27 directories above.
