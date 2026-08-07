# Any Advice on Cross-Validation Strategy?

- archived_at: 2026-06-11T13:50:14Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700827

Topic #700827: Any Advice on Cross-Validation Strategy?
  Author: NobelK
  Posted: 2026-05-18 06:48:51.684000
  Votes: 2  Comments: 3

I'm having a lot of difficulty coming up with a reliable cross-validation strategy for this competition.
If anyone has any helpful references, learning materials, or general advice on CV design, I would really appreciate it.
My intuition is that validation design is likely to be one of the most important keys to performing well in this competition.

Comments:
├─ PC Jimmmy (2026-05-18 17:08:08.527000) [+0]
│  The split I would recommend depends on your use of typewells.  A discussion post reported that a number of the typewells were duplicates.  
│  
│  That analysis was partially correct (assuming I did not ...
  ├─ hengck23 (2026-05-18 19:42:54.390000) [+0]
  │  Since you can see location and number of the hidden test wells from host explanation slides, create similar validation split based on that, mimicking the test distribution
├─ PatrickAIForFun (2026-05-18 07:07:26) [+0]
│  I would just go with standard Grouped K-Fold CV where the well id is the group. As per my understanding the hidden test set has wells very close to existing wells and also includes some typewells w...
