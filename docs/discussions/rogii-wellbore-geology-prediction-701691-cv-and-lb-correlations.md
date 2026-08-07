# cv and lb correlations .....

- archived_at: 2026-06-11T13:48:36Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/701691

Topic #701691: cv and lb correlations .....
  Author: Gaurav Rawat
  Posted: 2026-05-19 01:24:38.821000
  Votes: 12  Comments: 16

Was seeing some notebooks getting better in LB based on the cv , So far wanted to check how others have been getting the correlation going for tabular and non tabular models . For me have been using the standard GroupKfold on wells . 

Train Log 




Version
CV RMSE (ft)
LB RMSE (ft)




train_v2.py
31.3871
35.843


train_v2.1.py
14.7065
13.949


v2.2
14.4634
13.777


v2.5
11.9993
12.383


v2.6
11.2693
12.383


v2.7
10.7485
10.606


v2.7.1
10.2486
-


v2.8
10.6543
-


v2.7.2 online
-
10.520


v2.9
10.3256
9.816


v2.10
10.3730
10.384


v2.9.7
10.37
9.585


v2.9.11
10.6
8.739

Comments:
├─ Ruby (2026-06-08 14:30:58.543000) [+5]
│  recent two experiments:
│  CV 6.74 LB 6.48
│  CV 6.22 LB 7.18
│  I guess it is dominated by some bad cases
├─ Tucker Arrants (2026-06-08 14:23:48.600000) [+3]
│  LB feels noisy to me. I often observe CV improvements of 0.7 feet or more, leading to regressions in LB. I think some of my submissions were "lucky" e.g. CV around 8 scoring 6.6 on LB.
│  
│  Latest resu...
  ├─ Jack (2026-06-08 19:34:22.513000) [+0]
  │  I'd be questioning where those CV improvements are coming from in relation to past runs - might be insightful. I'm still trying to figure out what the heck you're doing for 2 mins inference.. well,...
    ├─ Gaurav Rawat (2026-06-08 19:49:29.430000) [+0]
    │  I dunno CV strategy needs to be alteare in one I got cv 7.4 but LB is like 9 . maybe need to have cv mixed with hard wells vs easy ones per fold or some custom way
    ├─ Jack (2026-06-08 20:10:27.660000) [+0]
    │  But how would you define hard vs easy wells? I can think of some more direct ways of balancing folds
├─ Tucker Arrants (2026-05-29 02:50:39.137000) [+6]
│  Single model NN update:
│  
│  CV 8.5, LB 7.5
│  
│  Inference in 2 minutes lol
  ├─ Gaurav Rawat (2026-05-29 03:31:37.767000) [+1]
  │  awesome ya I see NN infer like 2-3 mins .. :) maybe u framed the right Arch .. my cv not going down
├─ Gaurav Rawat (2026-05-28 17:01:38.753000) [+1]
│  Adding NN experiments now , just baselines now
│  
│  
│  
│  CV 14.4 LB 17
│  
│  cv 8 lb 9
├─ Tucker Arrants (2026-05-19 02:03:43.993000) [+5]
│  With the plain jane GBDT models, CV around 11.00 split on well ID and leaderboard around 9.6
│  
│  Large gap, but very stable -&gt; all CV improvements have led to LB improvements (so far).
│  
│  A lot of th...
  ├─ Gaurav Rawat (2026-05-19 02:07:19.763000) [+1]
  │  feel need to beat the 10 mark in cv to see marked improvements for my experiments . NN so far for me havent been doign that great maybe need to dive deep to design them better .
├─ shanzhong8 (2026-05-21 05:32:43.863000) [+1]
│  CV 10.7  , LB 9.9
  ├─ Gaurav Rawat (2026-05-22 03:03:57.753000) [+0]
  │  nice GBDT or NN .. was wondering how much folks are gettign with NN CV
    ├─ shanzhong8 (2026-05-23 08:53:31.673000) [+2]
    │  Transformer
├─ Hassan Gasim (2026-05-20 06:22:27.067000) [+-3]
│  Outstanding progress so far. Regarding your note on Neural Networks underperforming: standard MLPs usually struggle with the spatial, sequential nature of wellbore data compared to GBDTs. Since wel...
├─ Durga Kumari (2026-05-19 15:09:56.927000) [+-4]
│  Interesting that v2.10 had slightly worse CV but matched LB almost perfectly. Usually a good sign the model is generalizing more consistently rather than optimizing fold-specific patterns.
├─ YYH (2026-06-09 01:26:58.327000) [+0]
│  Are the top-ranked solutions currently all based on physics models combined with machine learning? It seems that some physics models have performed quite well in this competition.It is easy to find...
