# Over fitting on well

- archived_at: 2026-06-11T13:48:42Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698644

Topic #698644: Over fitting on well
  Author: ren toda
  Posted: 2026-05-11 02:35:11.309000
  Votes: 1  Comments: 1

I'm not sure but , does adding mean tvt or Max tvt and things with cause over fitting by identifying the specific well instead of leaving the relationship.
I think it's medecated my the modern model structure.( e.g. boosting model feature frac) But I wonder how much it effects.

--- Training Stats ---
Total Samples:  5,092,255
Total Wells:    773
Avg Samples/Well: 6,587.65

Comments:
├─ Aly Ayman (2026-06-08 20:22:06.310000) [+-3]
│  Hello, that's very good questions
│  
│  Yes, per-well aggregate features (mean TVT, max TVT, etc.) are a leakage/overfitting risk …. but the mechanism matters.
│  
│  There are two distinct problems in your q...
