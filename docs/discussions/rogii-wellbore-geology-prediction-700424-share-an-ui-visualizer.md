# Share an UI visualizer

- archived_at: 2026-07-02T13:56:36Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700424

Topic #700424: Share an UI visualizer 
  Author: Tom
  Posted: 2026-05-17 12:47:20.845000
  Votes: 79  Comments: 63

Feel free to use it: https://github.com/tom99763/rogii-viewer  

And some EDA &amp; directions in attachments. I suggest people to read glossary.html first to clarify some confused definition. 

I really like this explaination about TVT:

Analogy: TVT is the “floor number” in a geological building

Imagine the geology as a completed high-rise building, where each floor represents a different rock layer (ANCC, Austin Chalk, Eagle Ford, Buda, etc.).

Typewell = the elevator shaft: it goes vertically from the top floor to the basement, recording “the GR at this layer is X” as it passes through each floor.
Horizontal well = a person walking inside the building:  moving along the hallway of a certain floor, sometimes going up or down between floors.
TVT = which floor is this person currently on?


Update GR Mismatch Visualizer

Comments:
├─ Tom (2026-06-12 12:59:10.873000) [+2]
│  Even though this was 5 years ago, I try seasonal trends again.
│  
│  https://www.kaggle.com/code/tom99763/tensorflow-probability-inference?kernelSessionId=79007085
  ├─ Tom (2026-06-12 15:12:53.600000) [+0]
  ├─ hengck23 (2026-06-13 03:26:44.953000) [+4]
  │  i have a feeling that the interpretation is different for different wells (hence it is mixture mode). It is not a "optimization/prediction" problem but rather predicting the geologist preference fo...
  ├─ hengck23 (2026-06-13 20:24:15.563000) [+3]
  │  if you are predicting trajectories, here is an alternative formulation. very much like onion layering of the Vesuvius Challenge - Surface Detection.
  │  
  │  each value (or segment, e.g peak or valley) of ...
├─ hengck23 (2026-06-03 23:52:18.897000) [+5]
│  the public notebook (recent lb 8.63) uses meta-heuristics to decide k-beam or pf. this means some information is embedded at "whole well level", e.g. len of csv, min and max of well z, etc…
│  
│  proble...
├─ Tom (2026-06-03 12:58:36.763000) [+5]
│  Explaination about current trick and physical meaning in attachment
│  
│  And is it possible to do point sampling instead and making model predicting point connections. Just initial thoughts.
├─ Tom (2026-05-27 13:06:06.467000) [+5]
│  Working on Neural SDE now. I discover that forward-stepping curriculum can make it start to learn.
│  This is really badass.
├─ Tom (2026-05-23 05:48:45.080000) [+3]
│  Share another approach: curvature integration with teacher forcing warm start
├─ Tom (2026-05-23 01:45:41.363000) [+3]
│  Sharing a diffeomorphic warping approach from my vesiuvius challenge solution. (Warp from a flat line instead
  ├─ Tom (2026-05-23 02:03:20.400000) [+1]
  │  So, building on Giba’s last-value baseline, a promising next step is to predict the direction at each MD position using sign(x), and then iteratively refine the correction magnitude over N steps.
    ├─ Tom (2026-05-23 02:08:10.620000) [+2]
    ├─ hengck23 (2026-05-23 03:18:42.403000) [+2]
    │  i have a slightly different but similar idea:
    │  
    │  treat it like a RL game:
    │  
    │  input solution
    │  while some stop condition not meet:
    │  - analyse and select a segment
    │  - push the segment up or down
    │  - accept if ...
    ├─ Tom (2026-05-23 11:40:39.573000) [+2]
    │  Now I develop a piecewise correction model by defining multiple pieces using two split points, t1 and t2, along md.
    │  
    │  A piece is defined as:
    │  
    │  $$
    │  piece =
    │  sign(correction_{t1}^{oof} - model_{t1}^{oof}...
├─ Tom (2026-05-23 14:01:53.093000) [+4]
│  Probabilistic modeling is shining (No GBDT, finish in 20min)
  ├─ Tucker Arrants (2026-05-23 19:00:10.470000) [+8]
  │  ~1.2 ft behind you with simple UNet model. No physics constraints yet.
  │  
  │  Pre-training on synthetic wells gave a decent boost…
    ├─ Sangram Patil (2026-06-06 17:22:58.513000) [+2]
    │  How are you using the 2D U-Net? My input is [B, C, H, W] and the output is [B, H, W], but the model isn't performing well. I can't get the FT score below 14 no matter what I try. Do you have any su...
    ├─ Tucker Arrants (2026-06-08 02:15:08.823000) [+1]
    │  Your output has the right shape. Ask yourself what each of the H rows along a single column is competing to be and whether you're scoring that competition, or just regressing its shadow.
    │  
    │  Look at w...
    ├─ hengck23 (2026-06-09 21:31:42.040000) [+2]
    │  "you're scoring that competition, or just regressing its shadow."  
    │  
    │  within one column, SDF actually ranks all (typewell, horizontal) matches and gives results in "distance form". that is why it is...
    ├─ Sangram Patil (2026-06-10 05:00:43.733000) [+2]
    │  Thanks, guys, @tuckerarrants and @hengck23. I'm still confused about most parts, so I've been using Claude and Gemini to help me understand them. At least I managed to build an SDF baseline that ma...
    ├─ hengck23 (2026-06-10 05:14:11.403000) [+4]
    │  Nice work getting an SDF baseline running.
    │  
    │  One caution: “matches CV–LB” is useful but not a litmus test, but it is not enough to prove the implementation is correct.
    │  e.g. A wrong local CV and a wr...
    ├─ Tucker Arrants (2026-06-10 05:40:46.857000) [+3]
    │  Good start - keep trying and lean on the LLMs. It took me a little to get mine running, but it will give you a very good understanding of the problem / data once you do. I think there are a lot of ...
├─ hengck23 (2026-05-18 20:11:02.220000) [+5]
│  https://youtu.be/VgzFt7xknGo?si=rwz9Kv2oi3ZwniBE
│  
│  time 27:50 shows (results? or the infrence  process?) ROGII automatic alignment and  segmentation
├─ Tom (2026-05-21 16:35:22.610000) [+3]
│  about fourier formation perspective on this problem
├─ Tom (2026-05-20 15:13:20.400000) [+3]
│  On DTW inverse problem in the wavelet domain
  ├─ hengck23 (2026-05-20 18:23:38.070000) [+0]
  │  in the typewell GR domain would be better?
  ├─ Gaurav Rawat (2026-05-23 15:10:54.597000) [+0]
  │  Did dtw work for you
├─ Tom (2026-05-19 11:04:43.927000) [+3]
│  Update: 
│  
│  Baysian Physical-informed SegFormer got very good result.  (0.94 cv score)
│  
│  
│  
│  
│  Fold
│  baseline (077o)
│  soft_seg input (077q)
│  Δ q-o
│  bpinn (077bpinn) ⭐⭐
│  Δ bpinn-q
│  
│  
│  
│  
│  1
│  9.45
│  9.18
│  −0.27
│  9.18
│  −0...
  ├─ hengck23 (2026-05-19 11:17:04.340000) [+3]
  │  If you consider just gr fitting aline, transformer is clearly better. Current notebook has better results because it is a fusion of multiple methods.
  │  
  │  You should analyse 1. Comparison with last val...
    ├─ hengck23 (2026-05-19 13:11:43.413000) [+1]
    │  you can try to add the following to the transformer as features, it should improve  results by 1~2
    │  
    │  
    │  
    │  shared common typewell id
    │  
    │  x,y,z, amz, inc
    │  
    │  plane z  sampled from fitted geology plane  ancc, b...
    ├─ Simon Beck (2026-06-21 08:23:30.123000) [+0]
    │  @hengck23 now this is gold. i shoulde try it. thx
  ├─ NobelK (2026-05-21 18:08:32.407000) [+0]
  │  That's amazing!!
  │  How is this learning being done?
  │  I'm very curious about the details.
├─ Tom (2026-05-18 09:11:14.423000) [+4]
│  I made a NN-based approach with some probabilistic modeling similar to @jeroencottaar Yale/UNC-CH solution. I haven't submitted yet but it seems a good direction.
  ├─ PatrickAIForFun (2026-05-18 12:54:45.643000) [+3]
  │  When you say similar to Jeroen's solution, do you mean you actually modelled a prior and are optimizing the TVT math to minimize some measruement of mismatch between the TVTs? If yes, this is highl...
    ├─ Tom (2026-05-18 13:16:46.113000) [+4]
    │  Yes, I’ve actually built few priors and have been working on minimizing several measurements. I’ll release it once I have more completed experiments and the full map is built.
    ├─ hengck23 (2026-05-18 13:23:58.057000) [+3]
    │  ", but I don't see how a CNN would fit into this" . put your prior in the loss:  
    │  
    │  loss = regression loss + classification loss  + "too different from the prior loss"  
    │  
    │  if you use probability:
    │  too...
    ├─ Tom (2026-05-18 13:36:12.803000) [+1]
    │  I use this package: https://docs.pyro.ai/en/stable/
├─ hengck23 (2026-05-17 15:26:24.330000) [+3]
│  U can add following to cnn feature:
│  
│  
│  
│  Self correlation, good for identifying moving reverse 
│  
│  Neighbouring well correlation
├─ Tom (2026-05-17 13:37:40.257000) [+3]
│  This plot seems showing a hidden insight
├─ hengck23 (2026-05-17 15:33:51.527000) [+4]
│  Normal dtw assume monotonic seq and cannot match reverse index, so be careful if you use it.
├─ Tom (2026-05-22 00:55:42.623000) [+2]
│  single bpinn reach 9.+ lb
  ├─ Gaurav Rawat (2026-05-22 03:25:17.843000) [+1]
  │  Super what CV you getting ?
    ├─ Tom (2026-05-22 03:36:27.337000) [+0]
    │  9.62 cv score
    ├─ Gaurav Rawat (2026-05-22 21:20:49.623000) [+0]
    │  ahh awesome cv GBDTs are bit worse ,,,
├─ Tom (2026-05-19 07:53:52.450000) [+2]
│  Soft input segment worked. Just reach 0.97 overall CV with SegFormer. 
│  
│  
│  
│  
│  Fold
│  baseline
│  soft_seg input
│  Δ
│  
│  
│  
│  
│  1
│  9.45
│  9.18
│  −0.27
│  
│  
│  2
│  10.73
│  10.43
│  −0.30
│  
│  
│  3
│  9.44
│  8.68
│  −0.76 ★
│  
│  
│  4
│  11.39
│  10.48
│  −0.91 ★
│  
│  ...
├─ hengck23 (2026-05-19 06:32:27.897000) [+2]
│  Azimuthal LWD Data Interpretation for UBCTDGeosteering Using a Physics-Informed Neural Network
│  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6000576
│  
│  
│  
│  "stay / steer_up / steer_down" this cou...
  ├─ Tom (2026-05-19 07:01:27.140000) [+1]
  │  I think there's a opportunity to add some physical constraint in each segments for special behavior. Like controlling the curvature.
    ├─ Tom (2026-05-19 07:30:47.063000) [+1]
    ├─ hengck23 (2026-05-20 00:44:48.217000) [+1]
    │  if you can set some equations on
    │  
    │  forward GR =   F.interpolate (predict_tvt, typewell_tvt, typewell_GR)
    │  
    │  besides l2 loss, how to compare forward GR  and observation horizontal GR
    │  
    │  or 
    │  
    │  forward geol...
├─ hengck23 (2026-05-18 14:06:14.800000) [+2]
│  i have a suggestion for you. use seq transformer to learn segments (span of md of similar dip) and output as:
│  
│  segmentation output:
│  1111222222333333333333334444444444444444455666666666666666
│  
│  auxil...
  ├─ Tom (2026-05-19 05:47:19.473000) [+4]
  │  This is what a casual segformer can achieve for me now
    ├─ Mohit (2026-05-19 19:17:47.993000) [+0]
    │  even that much is good
    ├─ Gaurav Rawat (2026-05-20 02:00:11.063000) [+0]
    │  Thats great
├─ hengck23 (2026-05-18 12:01:20.630000) [+2]
│  maybe this is useful for 
│  you: https://www.youtube.com/watch?v=fEf6i2A0jdo
│  https://www.youtube.com/watch?v=vQDbKR3NAlM  
│  
│  check the lecture from 00 to 05 etc
├─ Gaurav Rawat (2026-05-18 00:48:33.427000) [+2]
│  love the eda via claude here very nice to understand the comp ..
  ├─ Tom (2026-05-18 01:51:56.097000) [+3]
  │  This comp is really complicated. There are many details haven't been coveraged
    ├─ hengck23 (2026-05-18 05:09:43.720000) [+3]
    │  one of the few competitions left that humans must do the problem definition first before applying agent optimization
    ├─ Tom (2026-05-18 05:30:01.593000) [+1]
    │  Turning "plan mode" in Claude and carefully define the problem by myself worked well for me.
    ├─ Tom (2026-05-18 06:12:54.617000) [+3]
    │  Workflow (every new experiment):
    │  
    │  
    │  
    │  Restate the geological problem
    │  (physical reality + constraints + signals)
    │  
    │  Abstract it into a mathematical problem
    │  (alignment? inpainting? inverse problem? assig...
    ├─ Gaurav Rawat (2026-05-18 14:21:38.300000) [+0]
    │  I try to do nowadays grill me for it to also grill before advising ..
├─ Durga Kumari (2026-05-18 15:06:36.027000) [+-1]
│  This is incredibly helpful, especially the TVT analogy.
├─ Mohit (2026-05-18 14:20:14.570000) [+-1]
│  Great work out there also how is nn aproach used here?
├─ Navneet (2026-05-18 07:57:27.787000) [+-1]
│  Cool UI visualizer @tom99763
├─  (2026-05-17 19:12:14.420000) [+0]
