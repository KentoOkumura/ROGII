# Using test_df from my dataset, does it not work??

- archived_at: 2026-06-11T13:49:45Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703181

Topic #703181: Using test_df from my dataset, does it not work??
  Author: Moonimonster
  Posted: 2026-05-29 01:57:22.762000
  Votes: 2  Comments: 2

Hello, hope y'all enjoying this competition!!
I have small question for those of you having fun with this competition.
I am currently using my train_df, and test_df that are all preprocessed, stored in my dataset.
For some reason, my kaggle notebook works well but when i submit my submission file to score, it crashes with errors saying 'Submission score error', or 'Notebook threw exception'.
I then checked my submission file whether it had different numbers of rows, id, and so on.. but turns out it has really not much of difference from the sample_submission or the ones that was scored successfully.

So  I was wondering if using test_df from my dataset, not from /kaggle/input/competitions/rogii-wellbore-geology-prediction/test causes trouble.
If anyone knows about it, plz help..! I'm also currently dealing with this issue. so i'll keep updating

Comments:
├─ Chris Deotte (2026-05-29 02:31:04.683000) [+3]
│  We cannot use our local test_df. When we submit our code, all the test data gets replaced with the real test data. This is done so that we cannot look at the test data with our human eyes. Only our...
  ├─ Moonimonster (2026-05-29 03:04:56.873000) [+0]
  │  Yeah that really makes sense!
  │  Thank you!!
