# be careful if you are using transformer and bfloat16

- archived_at: 2026-06-11T13:48:51Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703344

Topic #703344: be careful if you are using transformer and bfloat16
  Author: hengck23
  Posted: 2026-05-30 05:26:30.043000
  Votes: 13  Comments: 1

i am puzzled why my CNN works and transformer doesn't.  

in fact, for debug, i make a "copy transformer" where input seq = target tvt.
then i realise it is bfloat16 causing all the problems, even though i have already normalised input by mean,std

below are prediction of "copy transformer"  using bfloat16 and float32.

Comments:
├─ water joe (2026-06-08 13:01:59.630000) [+0]
│  Could you please tell me why this is the case? The performance of bfloat16 is very poor here.
