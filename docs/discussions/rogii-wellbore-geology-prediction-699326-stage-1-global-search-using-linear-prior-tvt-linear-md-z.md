# stage.1 : global search using linear prior tvt = linear(md,z)

- archived_at: 2026-06-11T13:49:16Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699326

Topic #699326: stage.1 : global search using linear prior tvt = linear(md,z)
  Author: hengck23
  Posted: 2026-05-13 15:49:38.319000
  Votes: 18  Comments: 5

Next post: stage.2 iterative local search for refinement

Comments:
├─ hengck23 (2026-05-14 03:21:23.183000) [+4]
│  this is why prior (constraints) is important.
│  
│  lower GR fitting doesn't mean lower T
│  ** it is an inverse problem ! **
│  
│  You should learn the 2-parameter prior space using TVT RMSE, not GR RMSE.
│  
│  VT.
├─ hengck23 (2026-05-13 17:03:16.327000) [+1]
│  initalise with fitted line of md,z after PS and also using tvt_input. need to think of  a way to make it "smooth"
  ├─ hengck23 (2026-05-13 17:12:46.277000) [+1]
  │  top to bottom: typewell TVT-GR after PS, typewell TVT-GS, horizontal MD-smoothedGS showing forward and reverse, horizontal MD-smoothedGS showing TVT as color, horizontal TVT-smoothedGS
    ├─ hengck23 (2026-05-13 17:21:44.083000) [+3]
    │  gemini suggested this but i haven't tried:
├─ Franklin Gois (2026-06-03 01:37:42.410000) [+0]
│  @hengck23 Thank you!
