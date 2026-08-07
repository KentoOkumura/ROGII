# What is difference between 'TVT' and 'TVT_input' 

- archived_at: 2026-06-11T13:50:19Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700792

Topic #700792: What is difference between 'TVT' and 'TVT_input' 
  Author: houzeyu2683
  Posted: 2026-05-18 05:08:56.912000
  Votes: 1  Comments: 3

hello, 
I check the test/ folders, I find the XXXXX__horizontal_well.csv in the train/ too.
Then I check the 'TVT_input' and 'TVT' are the same, and there are zero missing value in 'TVT'.
Is it the data-leak because I can copy the 'TVT' from train/.
Or … did I miss somthing?

Comments:
├─ PatrickAIForFun (2026-05-18 06:30:45.483000) [+1]
│  It is not a data leak - during inference on the actual hidden dataset you will be presented with wells which do not exist in the train set. Thus during inference you want be able to copy TVT and al...
  ├─ houzeyu2683 (2026-05-18 07:41:50.820000) [+0]
  │  Thanks, so the problem is training on a pool of wells, then infer on another wells.
├─ PC Jimmmy (2026-05-18 05:42:03.850000) [+0]
│  It was a big miss.  Read a number of the discussion posts.  It can be a tiny bit confusing - it's not a data leak.
