# Difference in train and test files

- archived_at: 2026-06-11T13:49:29Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703532

Topic #703532: Difference in train and test files
  Author: Navya Bhat
  Posted: 2026-06-01 02:03:21.316000
  Votes: 1  Comments: 2

Hi organizers / fellow competitors,
I'd like to confirm a structural difference I'm seeing between the train and test files before I build my feature pipeline.
 In the training horizontal-well files, the columns are:
 MD, X, Y, Z, ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA, TVT, GR, TVT_input
 In the test horizontal-well files, the columns are only:
 MD, X, Y, Z, GR, TVT_input
 So the target TVT (expected to be hidden) and the six formation-top columns : ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA appear in train but are absent from
 test.
 My questions:



Is it correct that these 6 formation columns will not be available at inference time, and therefore should be treated as train-only (i.e., not usable
as model input features)?

Or is their absence specific to the provided sample test files, and the actual hidden test set will include them?
I want to make sure I'm not engineering features on columns that won't exist when the notebook is scored. Confirmation would help avoid a train/test
mismatch.
Thanks!

Comments:
├─ Tucker Arrants (2026-06-01 02:08:44.097000) [+1]
│  Correct, they are training only
├─ kkj333 (2026-06-01 09:21:07.723000) [+-1]
│  Thanks for confirming — that matches what we see in the sample files.
│  
│  For others building feature pipelines: at inference time the hidden test horizontal files only expose MD, X, Y, Z, GR, TVT_inp...
