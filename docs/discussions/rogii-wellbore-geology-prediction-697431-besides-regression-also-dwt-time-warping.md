# besides regression, also dwt (time warping)! 

- archived_at: 2026-06-11T13:49:53Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697431

Topic #697431: besides regression, also dwt (time warping)! 
  Author: hengck23
  Posted: 2026-05-06 03:38:07.141000
  Votes: 34  Comments: 36

related to geosteering. Here, given a strip of MD-GR of the horizontal well, "stretch and fold" it so that it matches the TVT-GR reference of the typewell







but, since the test and train locations are pretty close, pure regression might just also work

Comments:
├─ hengck23 (2026-05-24 13:53:22.063000) [+1]
├─ hengck23 (2026-05-18 14:00:49.023000) [+6]
│  surprise surprise surprise
│  there are only 69 unique typewells in train data
├─ hengck23 (2026-05-17 21:14:20.227000) [+4]
│  suddenly, i think of folding protein solution: alphafold. here we are essentially folding the horizontal GR signal. each possible trajectory is a confomer. typewell and neigbhours GR are "binding s...
├─ hengck23 (2026-05-16 01:58:36.620000) [+3]
│  Real-time forward modeling and inversion of logging-while-drilling electromagnetic measurements in horizontal wells
├─ hengck23 (2026-05-18 20:03:12.817000) [+1]
│  data augmentation
├─ Tom (2026-05-17 02:13:24.160000) [+1]
│  Any Physics-informed ML approaches discovered? @hengck23
  ├─ hengck23 (2026-05-17 02:21:45.807000) [+5]
  │  issue is not physics modeling. you can verify your results with forward differentiable model:
  │  
  │  error = observed GR - generated GR = typwell GR (torch inteploated tvt as lookup index)
  │  
  │  
  │  issue is inv...
├─ hengck23 (2026-05-16 18:34:00.233000) [+1]
│  neighbour can help! e,g, they tell you the range of horizontal tvt
  ├─ PatrickAIForFun (2026-05-16 19:58:40.200000) [+1]
  │  You are showing/comparing the typewell logs, correct? If yes, then this is expected an has already been found (although not shifted matches, but exact matches): https://www.kaggle.com/competitions/...
    ├─ PC Jimmmy (2026-05-26 15:42:49.200000) [+1]
    │  If you look for shifted matches I found only 57 type wells in the entire field.
├─ hengck23 (2026-05-17 10:00:33) [+2]
│  Someone should probe the hidden typewell to see if they are offset copies of train. I think some are. If they are, you have free geology infotmation copied from train
  ├─ PatrickAIForFun (2026-05-17 11:41:15.740000) [+1]
  │  I can neither confirm nor deny with full certainty, however based on my observations and testing it also seems very likely that the hidden test set shares some typewells with the public training se...
    ├─ hengck23 (2026-05-17 12:06:15.220000) [+1]
    │  you can check the host competition PPT. he shows the hidden test well location
    ├─ hengck23 (2026-05-17 12:07:36.930000) [+1]
    │  "apart from the geology labels of the typewell, this does not provide much more information" you are wrong.
    │  the model now become tvt = model(shared type well, known tvt, full tvt of neighbours (inc...
    ├─ Tom (2026-05-17 13:47:30.753000) [+2]
    │  With these information and referencing the typewells, you can even build a very strong transformer or GNN to encode them
    ├─ PC Jimmmy (2026-05-26 15:40:38.063000) [+0]
    │  hengck23
    │  
    │  When you checked the ppt did you end up with 45 test well paths?
├─ hengck23 (2026-05-16 02:07:27.100000) [+1]
│  https://www.rogii.com/blog/starsteer-geoassist-enhanced-eagle-ford-reservoir
│  ROGII implemented StarSteer's ML-based GeoAssist to automate geosteering.
│  
│  which parts are the most and least confident?...
  ├─ hengck23 (2026-05-16 02:20:46.317000) [+1]
  │  where are the typewells? how is the global dip related to the xy slant horizontal drill path?
  ├─ hengck23 (2026-05-16 02:28:29.603000) [+1]
  │  their heatmap looks very good (heatmap is some similarity between horizontal GR and reference geology?)
  │  
  │  
  │  
  │  https://www.rogii.com/blog/the-hidden-cost-of-switching-between-geoscience-tools
  │  how wells...
├─ hengck23 (2026-05-16 01:43:18.277000) [+2]
│  related:
│  
│  related:
│  https://github.com/hhschumann/LWD_inversion
│  "This project aims to use gamma ray loging while drilling (LWD) measurements to invert for the position of a geologic interval relativ...
├─ hengck23 (2026-05-08 04:46:07.447000) [+2]
│  plot in 3d and it is a folding problem
  ├─ hengck23 (2026-05-08 04:47:47.360000) [+1]
├─ hengck23 (2026-05-06 11:01:23.030000) [+2]
│  forward model?
│  
│  hfile = "0a57a29c__horizontal_well.csv"
│  tfile = "0a57a29c__typewell.csv"
│  h  = pd.read_csv(f"{KAGGLE_DIR}/train/{hfile}")
│  tw = pd.read_csv(f"{KAGGLE_DIR}/train/{tfile}")
│  tw_tvt = tw[...
├─ hengck23 (2026-05-06 10:59:03.640000) [+3]
│  the trick to winning is to "somehow" reconstruct the "3d geological site" using the train AND test data, since the wells are in the same "site"
├─ hengck23 (2026-05-06 10:51:46.880000) [+3]
├─ hengck23 (2026-05-06 03:51:13.910000) [+2]
│  https://github.com/luthfigeo/DTW-Stratigraphic-Correlation/blob/main/DTW.ipynb
├─ hengck23 (2026-05-09 18:17:55.963000) [+2]
│  deep net logit:  horizontal md length x location of reference (each location is a class)
│  
│  
│  
│  training iterations of the transformer:
│  
│  
│  
│  it figures out the best way is to grow from PS?  I actually ex...
├─ hengck23 (2026-05-09 05:43:53.623000) [+1]
│  plot of md vs dTVT
│  
│  
│  
│  
│  
│  this tells you how the ground truth is created … reverse engineering?
  ├─ hengck23 (2026-05-09 07:49:03.390000) [+2]
  │  why is dy constant? synthetic data????
├─ hengck23 (2026-05-07 04:52:16.537000) [+1]
│  piece-wise fitting DTW.
│  
│  model predict (start,end, dTVT/dMD slope) for each segment.
│  
│  but i think the original DP in cost matrix is better 
│  
│  
│  
│  
│  
│  
│  
│  When the drill moves a long distance in MD / XYZ, ...
  ├─ eugene (2026-05-07 10:59:17.863000) [+0]
  │  Do I understand you method correct? 
  │  You move a window along the hw_gr, after ps point. For each window, you find the closest window on tw_gr with DTW. Then you look at which TVT is closest to the ...
    ├─ hengck23 (2026-05-07 11:46:55.897000) [+3]
    │  It is not the normal dtw. The index can be reversed depending if the drillhead is travelling upwards or downward. The noise is quite large, maybe you need to restrict to local search.
    │  
    │  The host pos...
    ├─ hengck23 (2026-05-07 12:32:34.120000) [+0]
    │  @evgeny000
    ├─ eugene (2026-05-07 13:01:01.020000) [+0]
    │  Thanks for the explanation! I still don't fully understand the data yet 😬, I hope it will be more clear after watching the video you mentioned. As far as I understand, you don't use tw data? 
    │  
    │  The ...
    ├─ hengck23 (2026-05-07 13:09:09.677000) [+3]
    │  You should think of it like that: reference vertical typewell has gr that encodes the geologic location called tvt. We are in horizontal well with unknown location. We want to know what is our tvt....
├─ Navneet (2026-05-10 05:38:00.933000) [+0]
│  Cool info on geosteering @hengck23
