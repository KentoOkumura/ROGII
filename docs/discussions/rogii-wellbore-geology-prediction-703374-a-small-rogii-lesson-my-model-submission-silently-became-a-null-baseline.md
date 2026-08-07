# A small ROGII lesson: my “model” submission silently became a Null baseline

- archived_at: 2026-06-11T13:49:35Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703374

Topic #703374: A small ROGII lesson: my “model” submission silently became a Null baseline
  Author: Pacifista
  Posted: 2026-05-30 12:44:54.729000
  Votes: 1  Comments: 0

Hi everyone,

I wanted to share a small lesson from my early experiments in this competition, partly as a warning and partly to ask how others are thinking about validation.

At first, I tried a very simple safe baseline around last_anchor_tvt.

My rough results so far:

Method    Public LB / CV note
Naive slope blend    Public LB: 22.887
Null fixed last_anchor_tvt    Public LB: 15.883
XGB/CatBoost artifact inference    Public LB: 14.852
CatBoost-only artifact inference    Public LB: 14.940

The surprising part for me was how strong the simple Null baseline was.

In my validation, naive slope extrapolation looked quite dangerous. My interpretation is that the local TVT slope before the evaluation zone does not necessarily continue into the hidden interval, and when it fails, the error can grow quickly.

Another lesson was specific to Code Competition submissions.

One of my early “model” submissions got exactly the same score as the Null baseline. After checking the notebook path, I realized that the model inference had likely failed and silently fallen back to Null.

Since then, I started printing diagnostics like:

USED_FALLBACK
loaded_xgb_models
loaded_cat_models
feature_columns_match
diff_from_null_nonzero_count
NaN count
inf count

This helped confirm whether the submitted submission.csv was actually from the model or just the fallback.

For example, in the successful artifact inference run, I confirmed:

USED_FALLBACK=False
loaded_xgb_models=5
loaded_cat_models=5
diff_from_null_nonzero_count=14151
final submission source=model

That run improved my Public LB from 15.883 to 14.852.

My current thinking is:

last_anchor_tvt is a very strong baseline.
Unconditional slope extrapolation can be harmful.
GroupKFold by well_id seems reasonably aligned with Public LB for these simple baselines.
CatBoost is strong, but a small XGBoost blend helped more than CatBoost-only in my current setup.
Artifact/fallback diagnostics are essential for this Code Competition format.

My question for others:

How are you validating spatial or formation-based features without leakage?

For example, if using nearby wells or formation surfaces, my current thought is:

build KNN / spatial features only from train-fold wells,
apply them to validation-fold wells,
and only use all train wells when generating test predictions.

Does this sound like a reasonable setup, or are there better validation strategies for this kind of geological correlation problem?

I would also be curious whether others are seeing similar behavior with:

Null baseline strength,
slope extrapolation instability,
GR matching / cross-correlation features,
or hard wells dominating the error.

Thanks, and good luck everyone!

No comments
