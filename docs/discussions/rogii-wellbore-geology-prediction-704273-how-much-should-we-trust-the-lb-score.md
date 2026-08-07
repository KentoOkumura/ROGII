# How much should we trust the LB score?

- archived_at: 2026-06-11T13:49:09Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/704273

Topic #704273: How much should we trust the LB score?
  Author: 寿!
  Posted: 2026-06-04 04:06:20.449000
  Votes: 11  Comments: 6

I'm relatively new to this competition, but it seems there's some distribution shift between train and test. My local CV and public RMSE diverge by up to 2 in some cases. On top of that, which one is more optimistic (local vs. public) varies depending on the prediction approach. Methods that rely on specific assumptions or modeling tend to show a larger gap. In my case, a spatial method using offset wells gives CV &lt; LB (public is more pessimistic), while the particle filter approach that's been popular in recent notebooks gives CV &gt; LB (public is more optimistic). 
My thinking is that since the training set has 773 wells and the public test set only has 52, we should trust local CV over the LB, assuming our validation strategy is sound. 
What do you all think?

Comments:
├─ Jack (2026-06-04 22:08:41.320000) [+2]
├─ Ulrich G. (2026-06-04 09:01:19.163000) [+1]
│  I think we could trust, for the time being there is a line-up between CV and LB for me
  ├─ 寿! (2026-06-04 12:11:39.080000) [+1]
  │  That makes sense. I also feel like there tends to be a trend where LB improves when CV improves, though the ranges seem to be on different scales.
    ├─ Tucker Arrants (2026-06-04 15:50:34.567000) [+0]
    │  Yes, when I make larger pipeline changes, my CV-LB correlation "resets" but then any CV improvements with small tuning within that new pipeline always lead to LB improvements. 
    │  
    │  Perhaps has to do t...
├─ Tim Krige (2026-06-04 11:13:10.277000) [+2]
│  In my opinion, both are important. I think that leaderboard probing is a real risk here, and your comment of trusting local CV therefore has some merit, however, dataleaks are of critical importanc...
  ├─ 寿! (2026-06-04 12:15:39.953000) [+1]
  │  Thank you for the insightful advice! You're right that a statistically-grounded approach seems to be key here.
