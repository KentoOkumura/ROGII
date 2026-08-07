# Handling Rolling and Lag features for Testing data

- archived_at: 2026-06-11T13:48:20Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/704383

Topic #704383: Handling Rolling and Lag features for Testing data
  Author: Pratyaksh
  Posted: 2026-06-04 10:18:13.945000
  Votes: -1  Comments: 7

I have done my model training but i am puzzled on how to create the rolling and lag features for infrence  or test data . how are you guy doing it

Comments:
├─ PatrickAIForFun (2026-06-04 10:24:22.030000) [+0]
│  What exactly is your issue there? The test set at inference looks exactly the same as the trainung data (except for the train-only coljmns TVT, ANCC, …, Geology). Thus, the approach you used to gen...
  ├─ Pratyaksh (2026-06-04 10:33:17.383000) [+0]
  │  Maybe I'm misunderstanding something then.
  │  
  │  For example, suppose I create a GR_rolling_20 feature. In training I can choose to drop the first 19 rows (or use min_periods=1), but at inference time t...
    ├─ PatrickAIForFun (2026-06-04 10:45:27.503000) [+0]
    │  You only need to predict the TVT starting at the prediction start point (the point where TVT_input becomes NaN). Thus there is always enough history before this point which you can use but do not n...
    ├─ Pratyaksh (2026-06-04 10:57:46.197000) [+0]
    │  idk if we are having a misunderstanding or something but i am talking about test dataset . and how to estimate those feature in test data for submission not even talking about the training or evalu...
    ├─ Chris Deotte (2026-06-04 13:11:36.827000) [+0]
    │  I don't understand your question
    │  
    │  
    │    
    │  but at inference time the test well also starts at row 0 and has no prior history before that point.
    │  
    │  
    │  The train data and test data are exactly the same. Whate...
    ├─ Pratyaksh (2026-06-11 09:52:56.410000) [+0]
    │  wait, correct me if i am wrong . Like test data is similar to train data . So we don't need to predict TVT values for whole of test data we need to predict them after the NAN values starts? . If ye...
    ├─ PatrickAIForFun (2026-06-11 10:01:05.823000) [+0]
    │  Exactly, you only need to predict tge rows where TVT_input is NaN. The rows before this predictuon start are already given.
