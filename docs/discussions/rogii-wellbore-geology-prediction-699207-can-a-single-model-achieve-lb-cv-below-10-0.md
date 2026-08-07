# Can a single model achieve LB/CV below 10.0?

- archived_at: 2026-06-11T13:50:02Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699207

Topic #699207: Can a single model achieve LB/CV below 10.0?
  Author: NobelK
  Posted: 2026-05-13 05:11:56.953000
  Votes: 12  Comments: 10

Hi everyone,

I would like to ask whether it seems possible to achieve a score below 10.0 on the LB or local CV using a single model.

At the moment, my single CatBoost model has plateaued. I am using mostly tabular features with CatBoost, and my local CV and LB are no longer improving much after basic parameter tuning. I suspect that either my validation strategy or feature engineering may be the bottleneck, but I am not sure how to diagnose it.

Since I am still a beginner, I would really appreciate any discussion or exchange of ideas around this topic.

For example, I am interested in hearing about:



whether people have seen single-model scores below 10.0

whether CatBoost alone seems strong enough for this competition

which direction is more promising: feature engineering, validation design, post-processing, or ensembling

common mistakes that may cause a CatBoost baseline to get stuck


I am not asking for anyone’s private solution, but I would be grateful for any general hints, observations, or advice that could help beginners understand where to focus next.

Thanks in advance!

Comments:
├─ Vishal Kishore (2026-05-24 10:45:49.383000) [+4]
│  Yeah it is possible, I scored 8.8 with a single dl model approach only. Only matters how you formulate it in your model
  ├─ NobelK (2026-05-24 14:26:39.577000) [+0]
  │  That's a fantastic score!
  │  
  │  I have a basic question: what does it mean to "formulate it in your model"?
├─ Andrew Lukyanenko (2026-05-20 09:54:33.780000) [+3]
│  This is definitely possible. I got 9.463 with a single model.
  ├─ NobelK (2026-05-20 10:02:44.883000) [+0]
  │  9.463 with a single model!? That's amazing!
  │  I'm very interested in your approach.
  │  I've run out of ideas right now…
├─ Tom (2026-05-17 08:12:29.687000) [+1]
│  Current GBDTs are definitely not a good solution for this challenge. Based on my current EDA, there is still significant room for improvement. Ultimately, it depends on how the problem is reformula...
  ├─ hengck23 (2026-05-17 08:46:35.987000) [+1]
  │  my experment results: upper bound 3.5 (due to ambigous annotation). i think a good model is about 4.5. 
  │  GBDTs is ok if the input is good. 
  │  
  │  but problem is not feature engineering, it is problem for...
    ├─ NobelK (2026-05-17 08:51:32.527000) [+1]
    │  I'd like to use CNNs and transformers, but I lack the knowledge to build a model properly.
    │  
    │  I would be grateful if you could share any helpful resources or resources you know of.
├─ Tucker Arrants (2026-05-13 11:37:33.610000) [+2]
│  Of course, the competition has only been live for a week. I have a simple LGB that scores 9.7 on the leaderboard - nothing fancy, just some feature engineering. I’m sure the final scores will be mu...
  ├─ NobelK (2026-05-13 12:12:38.163000) [+1]
  │  9.7… That's fantastic.
  │  
  │  I still seem to lack the fundamentals, so I'll do my best to catch up.
  │  
  │  Thank you for the helpful information; let's both do our best.
├─ Yang Wei Hao (2026-05-13 07:45:38.477000) [+-1]
│  i think it must do that
