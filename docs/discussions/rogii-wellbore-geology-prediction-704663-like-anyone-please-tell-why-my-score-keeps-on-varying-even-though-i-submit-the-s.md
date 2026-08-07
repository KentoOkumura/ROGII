# Like anyone please tell why my score keeps on varying even though I submit the same notebook?

- archived_at: 2026-06-11T13:49:05Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/704663

Topic #704663: Like anyone please tell why my score keeps on varying even though I submit the same notebook?
  Author: Debatreya Biswas
  Posted: 2026-06-05 17:00:25.712000
  Votes: 5  Comments: 6

like this is is the notebook I am talking about https://www.kaggle.com/code/debatreyabiswas/wellboregeology-prediction-with-koolbox-best-8-188

I submit it 3 times and all the 3 times different scores were given

like first time 8.354

second time 8.188(the best one)

third time 8.438

like is it because of the noise in the data we are using?

like I know the difference is only around 0.2 but when the models in the leaderboard board are only separated by 0.01 in some cases it becomes a huge deal, so it is just luck? Or is there a legit solution to this problem other than hoping for the best

(Sorry, I am new to Kaggle so the question might be stupid)

Comments:
├─ Chris Deotte (2026-06-05 21:10:30.897000) [+2]
│  Many feature engineering are stochastic in this competition. Feature engineering is the process of us making new columns on the train and test data. So every time we submit our notebook, the train....
  ├─ Debatreya Biswas (2026-06-05 21:32:50.960000) [+1]
  │  Ok,thank you Sir,
  │  
  │  So,the stochastic is the model randomness,(like for example while making a cake there is a instructions given add a pinch of salt in my code,so whenever I submit it,kaggle add di...
    ├─ Chris Deotte (2026-06-05 22:17:50.310000) [+5]
    │  No random is like random. In other tabular data competitions, there are no random features.
    │  
    │  In this competition we do things like particle filter features. To create a new column we literally gene...
    ├─ Debatreya Biswas (2026-06-05 23:00:09.223000) [+1]
    │  Wow,ok understood Sir🤯
    │  
    │  So,everytime we are submitting it is like a full on random roll of dice to generate the new column.
    │  
    │  So, basically there can exist a infinite possible number between my give...
    ├─ Chris Deotte (2026-06-05 23:04:44.600000) [+3]
    │  Yes exactly. A few weeks ago, one could get top 5 on the public LB (not private LB) just submitting the best public notebook like 10 times haha. But not anymore because the people at the top of the...
    ├─ Debatreya Biswas (2026-06-05 23:18:51.757000) [+0]
    │  Lol 😂 just praying for luck to get the top spot
    │  
    │  Thank,you Sir so much for the explanation and helping me understand it all
    │  
    │  Btw, Sir as you said the people at the top are now doing things differen...
