# Is online learning / test-time fine-tuning allowed?

- archived_at: 2026-06-11T13:50:26Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698002

Topic #698002: Is online learning / test-time fine-tuning allowed?
  Author: Kh0a
  Posted: 2026-05-08 04:09:03.521000
  Votes: 17  Comments: 5

With the current evaluation setup, participants can train the model offline and then fine-tune it using the test dataset at submission time by peeking at the actual test dataset. They can only extract the available TVT_input (calibration reference) and other features but can make a solid improvement through domain adaptation.

Specifically, at submission time in a Kaggle notebook:



Load the hidden test data (wellbores + GR, formation parameters, TVT_input)

Fine-tune the pre-trained model using test features as input (self-supervised learning using TVT_input as calibration targets)

Generate final predictions on the adapted model


Is this approach considered allowed under the competition rules? Looking forward to feedback from organizers.

Comments:
├─ Tucker Arrants (2026-05-14 20:37:59.090000) [+2]
│  Thank you for sharing your augmentation + online training technique. I consistently get about a 0.15 - 0.2 improvement from it. 
│  
│  Most recent training run:
│  
│  5 fold LGB without -&gt; 9.812
│  
│  5 fold L...
  ├─ Kh0a (2026-05-15 02:01:06.993000) [+1]
  │  Well done, although i am not sure if this is allowed yet.
├─ Kh0a (2026-05-08 15:49:55.707000) [+2]
│  I have tested with same training setup:
│  
│  online training: 10.953
│  
│  no online training: 11.323 
│  
│  The features processing idea was from https://www.kaggle.com/code/shinyanagai123/triple-signal-beam-se...
├─ hengck23 (2026-05-16 17:55:59.460000) [+0]
│  it should be allowed. Such methods had been used in previous kaggle competitions before.
│  it also includes things like
│  
│  
│  
│  finding statistic  like mean, std of test data  
│  
│  creating multiple window t...
├─ PC Jimmmy (2026-05-16 15:42:41.920000) [+0]
│  Test time learning has been an acceptable method for all the years I have been on kaggle.  The only issue I ever had was the compute time limit - in some past competitions my methods were too slow ...
