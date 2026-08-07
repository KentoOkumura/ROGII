# Submission Scoring Error — Is the scorer live yet?

- archived_at: 2026-06-11T13:49:34Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697329

Topic #697329: Submission Scoring Error — Is the scorer live yet?
  Author: Abdessamed Zetroni
  Posted: 2026-05-05 19:23:46.344000
  Votes: 8  Comments: 30

Hi, I've been trying to submit but keep getting a "Submission Scoring Error". My submission file matches the sample_submission.csv exactly 14,151 rows, correct id and tvt columns, no NaN values, all IDs verified against the sample.

Since this competition just launched and the leaderboard is empty, I'm wondering if the scorer isn't active yet on Kaggle's side. Can the organizers confirm whether submissions are open and the evaluation script is live?

Thanks

Comments:
├─ Ryan Holbrook (2026-05-05 22:12:50.100000) [+3]
│  Apologies for the scoring problems. The issue should be resolved now. Please let us know if you continue to have trouble or if anything else comes up.
  ├─ Santiago Maniches (2026-05-05 22:27:27.450000) [+0]
  │  Thank you! Working on it :)
  ├─  (2026-05-07 17:45:18.867000) [+0]
  ├─ Gideon Adom Boateng (2026-05-11 08:49:45.470000) [+0]
  │  Still facing scoring issues. My notebook successfully rans and generates the submission.csv but the scorer always fails this is my fifth time and the issue still persists. Kindly help out.
├─ parijit (2026-05-07 11:26:32.837000) [+1]
│  Hi
│  I dont see any errors when submitting the file however i have problems during the scoring process. The scoring process is still running and it has been a bit over 4 hours, that the scoring has n...
  ├─ Gideon Adom Boateng (2026-05-07 12:46:50.050000) [+0]
  │  Facing similar issue here
├─ Santiago Maniches (2026-05-05 21:31:32.903000) [+1]
│  Same situation here. Can you please check? Thank you!
├─ inversion (2026-05-05 20:22:32.520000) [+1]
│  Investigating . . .
├─ takaito (2026-05-05 20:00:14.760000) [+1]
│  I also failed to submit it.
├─ Chris Deotte (2026-05-05 19:56:00.617000) [+1]
│  I agree something seems wrong. I have tried multiple times to get my XGB starter to submit but i keep getting scoring errors. 
│  
│  Has anyone tried to just submit the sample submission? If that fails,...
  ├─ Chris Deotte (2026-05-05 20:06:45.197000) [+3]
  │  I tried submitting the sample submission, it failed here
  │  
  │  Admins, can you fix the LB? Thanks @addisonhoward @inversion @sohiermse
  │  
  │  UPDATE: Fixed
    ├─ Abdessamed Zetroni (2026-05-05 20:08:58.640000) [+1]
    │  Thanks for the validation Chris
├─ Pavlo Ivanin (2026-05-05 21:52:35.293000) [+2]
│  Yes, I also failed to submit
├─ Sean (2026-05-31 02:07:42.907000) [+0]
│  I am also facing the Submission Scoring Error as of May 30. The column names, row numbers and so on are exactly the same as the sample submission file though. Is there someone who hits the same pro...
├─ POR160893 (2026-05-08 12:11:08.410000) [+0]
│  Hello again,
│  
│  I wanted to provide a more detailed update because I am STILL consistently getting “Submission Scoring Error” despite extensive debugging and validation.
│  
│  At this point I do not belie...
├─ Navneet (2026-05-08 07:52:49.770000) [+0]
│  Thanks for the information on Submission Scoring Error @abdessamedzetroni
├─ Gideon Adom Boateng (2026-05-07 12:45:59.430000) [+0]
│  Same issue here. Code was able to generate the submission.csv but I could not get a score because it failed. Kindly help out
├─ POR160893 (2026-05-07 11:22:39.343000) [+0]
│  Hi everyone,
│  
│  I’m still experiencing persistent “Submission Scoring Error” issues in this competition after making 11 submissions up to this stage and wanted to provide a detailed summary of everyt...
  ├─ Ryan Holbrook (2026-05-08 11:47:38.907000) [+0]
  │  Hi @por160893,
  │  
  │  The scoring issue should be fully resolved now; it was only present for a few hours after launch. Have you tried your minimal submission that just submits the sample_submission.csv ...
    ├─ POR160893 (2026-05-08 13:07:47.580000) [+0]
    │  Hi @RyanHolbrook and organizers,
    │  
    │  I am still encountering a submission scoring error even after rebuilding the notebook from scratch and validating the submission file extensively.
    │  
    │  What I tested:
    │  ...
    ├─ Ryan Holbrook (2026-05-08 14:47:10.790000) [+1]
    │  Hi @por160893,
    │  
    │  
    │    
    │  I generated my actual predictions in RStudio locally and uploaded the CSV to Kaggle as a dataset.
    │  
    │  
    │  In this competition, you need to submit a notebook that creates your submissi...
├─ Nikita Shevyrev (2026-05-07 05:18:24.297000) [+0]
│  Hi everyone,
│  
│  I’m trying to understand the submission behavior for this code competition.
│  
│  Yesterday I had 5 failed submission attempts. Today I launched a new submission and noticed what looks lik...
  ├─ Ryan Holbrook (2026-05-08 11:50:38.467000) [+0]
  │  There are two runs kicked off when you submit from the notebook editor. One of them is a run on the published data that you see on the Data page. The other is the rerun on the hidden test data; tha...
├─ Chattso-GPT (Yasuhito Yanagisawa) (2026-05-07 01:16:40.407000) [+0]
│  I couldn’t even submit because of an error lol
  ├─ Chattso-GPT (Yasuhito Yanagisawa) (2026-05-07 16:56:54.843000) [+0]
  │  I’ve been stuck with this error all day and can’t get anything done… even in other competitions.
    ├─ Ryan Holbrook (2026-05-07 17:10:20.680000) [+0]
    │  Where are you seeing this error and what were you doing at the time?
    ├─ Chattso-GPT (Yasuhito Yanagisawa) (2026-05-07 17:52:03.403000) [+0]
    │  I always get this error whenever I try to create or operate a notebook. It’s been happening constantly since yesterday.
    ├─ Ryan Holbrook (2026-05-08 17:04:45.627000) [+0]
    │  I'm almost wondering if this is an authentication issue or something to do with a stale cache. Maybe try logging out and logging in again, or try logging in under an incognito session and see if th...
    ├─ Chattso-GPT (Yasuhito Yanagisawa) (2026-05-09 17:07:39.793000) [+0]
    │  Re-logging in didn’t help, but using an incognito session fixed it. Thank you very much!
├─ Bryce Chambers (2026-05-07 00:45:07.830000) [+0]
│  Can we upload just a .csv file?  Or is command line only required?
