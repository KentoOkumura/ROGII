# Notebook Threw Exception on submission despite successful local run

- archived_at: 2026-06-11T13:48:27Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/705417

Topic #705417: Notebook Threw Exception on submission despite successful local run
  Author: Alexander Osorio
  Posted: 2026-06-10 00:37:20.573000
  Votes: 1  Comments: 4

My notebook runs successfully (729s, generates submission.csv with 14151 rows) but every submission attempt shows "Notebook Threw Exception". The notebook logs show green checkmarks and correct output. This has happened across 5+ versions. Is there a known issue with submissions for this competition?

Comments:
├─ OpPrime (2026-06-10 07:45:38.033000) [+1]
│  I would run it cell by cell, also make sure you have not muted warnings so you can see what pops up.
│  
│  Then also look at if you are using P100s, and if you are getting a torch sm_60 accelerator erro...
├─ PC Jimmmy (2026-06-10 16:04:43.847000) [+0]
│  If you cannot figure it out - best advice-  make the notebook public and put a link in this discussion.  It's very hard to troubleshoot code you cannot see :)
│  
│  Had there been a link it's likely som...
├─ PC Jimmmy (2026-06-10 01:08:51.040000) [+0]
│  More likely issue is that your getting a memory error or shape error when the real test data is used.   The 3 fake test wells supplied might not be enough to stress your code.  Once the full (200 m...
  ├─ Chris Deotte (2026-06-10 14:56:39.287000) [+0]
  │  I agree. Probably a memory error. You can find it my using your inference notebook to infer 200 train wells. That will simulate what your inference notebook does during submit.
