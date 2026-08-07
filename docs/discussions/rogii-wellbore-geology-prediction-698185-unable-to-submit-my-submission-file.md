# Unable to submit my submission file

- archived_at: 2026-06-11T13:48:30Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698185

Topic #698185: Unable to submit my submission file
  Author: Chesang Irine
  Posted: 2026-05-08 19:37:33.793000
  Votes: 1  Comments: 7

I have downloaded my CSV file and am ready to submit it, but I’ve noticed that the upload button on the prediction submission page is not active. What should I do in this case? Also, whenever I try the quick option of “Save &amp; Commit,” the submission keeps failing.

Comments:
├─ Aly Ayman (2026-06-08 20:15:02.013000) [+-2]
│  Direct CSV upload is disabled for this competition by design. Submissions must come from a committed notebook that produces a file named exactly submission.csv. So instead of uploading, you'll subm...
├─ Olena Arshynnikova (2026-06-09 23:35:18.043000) [+0]
│  I can not submit my submission file (submit button is inactive). My notebook is turned off from the internet, it has a submission file (csv format). When I re-run all cells of my notebook, everythi...
  ├─ PC Jimmmy (2026-06-10 01:16:21.897000) [+0]
  │  Save Version - close the notebook.   Depending on your code it should be done in a few minutes as it's only predicting on the 3 fake wells.
  │  
  │  Go back to the notebook - Output tab - IF you only have ...
├─ PatrickAIForFun (2026-05-09 14:39:54.433000) [+0]
│  This is a code competition. Thus you must submit the notebook itself and not the csv.
│  This notebook is then re-run automatically with many more and different files in the test folder -&gt; this ens...
├─ PC Jimmmy (2026-05-09 00:01:40.583000) [+0]
│  The words "downloaded csv file" - you must generate the csv file in a kaggle notebook.   Where did you create the file?
  ├─ Chesang Irine (2026-05-09 12:52:01.540000) [+0]
  │  from my kaggle notebook…tried saving it many times  /saving the version but kept failing so i decided to download from the kaggle output files with intentions of uploading it manually only to reali...
    ├─ PC Jimmmy (2026-05-09 15:28:46.763000) [+0]
    │  You cannot upload it manually for this type of competition.
    │  When you run your kaggle notebook does the submission.csv file that was created look similar to what has been shared on all the code note...
