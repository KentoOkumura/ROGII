# Is the public LB test set (26%) fixed? 

- archived_at: 2026-06-11T13:49:07Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/701995

Topic #701995: Is the public LB test set (26%) fixed? 
  Author: Alhasan Abdellatif
  Posted: 2026-05-20 15:55:07.098000
  Votes: 7  Comments: 13

This leaderboard is calculated with approximately 26% of the test data. The final results will be based on the other 74%, so the final standings may be different.


I have noticed that running &amp; submittin the same exact notebook, same trained models, gives very different score with differences reaching over ~ 0.5 ft.  what does this mean? Is the public LB test set (26%) fixed?

Comments:
├─ Chris Deotte (2026-06-05 21:11:38.093000) [+4]
│  The test data does not change.
│  
│  The reason our scores change is because many feature engineering are stochastic in this competition. Feature engineering is the process of us making new columns on t...
├─ Zhenyu Zhang (2026-05-21 13:13:24.200000) [+-1]
│  I think it is not fixed
├─ Hamza (2026-05-23 10:56:46.810000) [+0]
│  I run my same LGB Baseline 2 times, 1st time I got score of 9.964 and Second time I got the score 9.477
├─ YtLiu (2026-05-22 06:51:17.183000) [+0]
│  The test set should be fixed. The score discrepancy you’re experiencing is most likely due to randomness in your code not being fully controlled. Multi-process parallelism and GPU usage can both in...
├─ PC Jimmmy (2026-05-20 17:48:37.077000) [+0]
│  As noted by PatrickAIForFun - the test has never varied in the 8 years I have been here.
│  
│  Not sure I understood your difference value - what is the smallest and largest score you have for what you ...
  ├─ Alhasan Abdellatif (2026-05-20 19:11:28.777000) [+0]
  │  For example, copying and re-submitting this top scored public notebook https://www.kaggle.com/code/nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based/notebook led to a 9.724 whic...
    ├─ PC Jimmmy (2026-05-21 12:30:01.137000) [+0]
    │  Copied and re-submitted the notebook and will let you know in few hours how it scored for me.  But as noted results do vary even with a very detailed seed, but 0.5 does seem a bit on the high side....
    ├─ PC Jimmmy (2026-05-21 15:00:40.743000) [+1]
    │  WOW - I did even worse at 10.146.
    ├─ PC Jimmmy (2026-05-21 15:09:17.487000) [+0]
    │  My LGB model rmse values match the posted original code.
    │  My Catboost values also match the posted orginal code.
    │  My Running Hill Climbing values match.
    │  My predicted values for the fake test data don...
    ├─ PC Jimmmy (2026-05-22 14:42:33.990000) [+0]
    │  In the original notebook that you forked the code from there is a comment from at least one other person who got a different value despite using the exact code.
├─ PatrickAIForFun (2026-05-20 16:15:43.110000) [+0]
│  This most likely means that not all randomness is fixed within your notebook (sometimes, even fixing all random seeds is not deterministic when using the GPU).
│  The 26% public split is fixed (not a ...
  ├─ Alhasan Abdellatif (2026-05-20 19:06:18.230000) [+0]
  │  Completely agree. It does not make sense if it vaires. I will double check the randomness in the notebook. Thanks!
    ├─ Radmir Zosimov (2026-05-21 13:28:41.353000) [+0]
    │  I had the same issue, it’s most likely your feature generation includes randomness, fix your seed. Also if you use numba seed has to be set inside a function
