# multi-trajectory prediction (MTP) with deep CNN for welllog inversion

- archived_at: 2026-06-11T13:48:53Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853

Topic #699853: multi-trajectory prediction (MTP) with deep CNN for welllog inversion
  Author: hengck23
  Posted: 2026-05-15 13:49:18.283000
  Votes: 47  Comments: 92

example notebook
https://www.kaggle.com/code/hengck23/cnn-mtp-example?scriptVersionId=320093395

arvix paper: "Direct Multi-Modal Inversion of Geophysical Logs Using Deep Learning" - Sergey Alyaev
https://arxiv.org/pdf/2201.01871
https://nfes.org/assets/workshop2022/ambrus_sequential_multi_mode_inversion_poster.pdf

[2D Heatmap Input] ──&gt; [Regression Head (CNN)] ──&gt; [MDN Predictor (MLP)] ──&gt; [Multi-Trajectory Output]

example of heatmap is shown below.
The Mixture Density Network (MDN) Predictor : for multiple paths hypothesis (like k-beam).

if you can identify match keypoints in GR signals, then you decide how to move/traverse between the matched keypoints.

Comments:
├─ hengck23 (2026-06-06 20:20:17.057000) [+3]
  ├─ hengck23 (2026-06-07 07:25:49.107000) [+1]
  │  CNN+SDF +MTP:   
  │  
  │  
  │  
  │  top3 prob path and last one is mean of top4+5   
  │  
  │  
  │  for the first time, we have correct prediction inside the top 3. 
  │  input is 512 window of h tvt (compression=2) and 64 window o...
    ├─ hengck23 (2026-06-07 07:29:55.810000) [+0]
    │  results of training mixture
    │  
    │  
    │  
    │  i think it has to be a mixture model because I can see the prediction hopping around a few modes
    ├─ Tom (2026-06-07 12:42:31.467000) [+0]
    │  They look more like some basis
    ├─ hengck23 (2026-06-07 15:03:27.597000) [+2]
    │  validation results for full length  tvt (by probing, all hidden test well has kength &lt;12_000). h tvt  window = 384, h tvt window 768 (at compression =16)  
    │  
    │  you can see  sdf is bending correctly...
    ├─ hengck23 (2026-06-07 23:28:50.513000) [+0]
    │  @tom99763 
    │  
    │  they are less primitive after plotting at the recovered TVT
    │  sdf = t_tvt - h_tvt, hence h_tvt = t_tvt - sdf  for sdf.abs()&lt;2 
    │  
    │  instead of generating more K per model, it is better to ...
├─ hengck23 (2026-06-03 07:12:22.640000) [+2]
│  The PF code uses lookup geo plane for training wells. what if we model the geo surface using grid interpolation? a validation rmse of 11.09 (non optimized)
  ├─ PatrickAIForFun (2026-06-03 11:24:17.303000) [+0]
  │  Yes, I can confirm - basic / non optimized kriging of geology layers gave a local RMSE ~11 and test rmse ~13.5 for me.
    ├─ hengck23 (2026-06-03 11:50:26.133000) [+0]
    │  Try better offset adjustment. Plot the graphs. Validation rmse should be near 11. Use the tvt input, geo predict and given well z near ps to determine best offset
├─ hengck23 (2026-06-01 05:29:45.583000) [+3]
│  transformer MTP of the previous post. I just need a good verifier
  ├─ hengck23 (2026-06-01 06:03:26.827000) [+0]
  ├─ sleep3r (2026-06-01 10:44:00.597000) [+0]
  │  gr matching is ill-conditioned: even at the true tvt horizontal &lt;--&gt; typewell gr only corr ~0.7, and offset error compounds
    ├─ hengck23 (2026-06-01 12:16:28.557000) [+0]
    │  The best way is to make a model that can recover h tvt from h gr = interp( h tvt, tw gr, tw tvt). This is perfect correlation but has multiple fp matches. If that works, you can introduce gausssian...
    ├─ hengck23 (2026-06-01 12:18:13.850000) [+0]
    │  My feeling is that we need to train a ranker or scorer rather than rely on generic correlation
    ├─ sleep3r (2026-06-01 12:26:55.423000) [+0]
    │  been down exactly this road
    │  
    │  mtp heatmap net + a learned catboost ranker over the modes (pairwise yetirank) + the h_tvt-from-h_gr recovery + gaussian/offset/scale/simplification aug
    │  
    │  two walls i co...
    ├─ hengck23 (2026-06-01 14:15:47.917000) [+0]
    │  we do not need to match all GR. we have good dtvt estimate. we just need a few anchor points to push the whole tvt curve to correct the pace.
    ├─ sleep3r (2026-06-01 14:29:27.077000) [+1]
    │  agreed a few anchors is all you need - with oracle anchors i get k≈10 down to ~4ft, k≈20 to ~1.7ft, so your pace-correction framing is right
    │  
    │  the catch is placing them: a local gr shift-search arou...
    ├─ hengck23 (2026-06-02 15:30:41.990000) [+3]
    │  @sleep3r 
    │  
    │  my suggestion is that you start with the native PF method from https://www.kaggle.com/code/sunnywu27/rogii-wellbore-tvt-physical-model  
    │  
    │  then replace the likelihood scorer with a learne...
    ├─ sleep3r (2026-06-02 19:25:44.183000) [+0]
    │  tried your cnn-likelihood idea pretty hard
    │  
    │  window matcher (h_gr vs typewell, learn the sdf) + noise aug, fold-safe. couldn't get it to beat the plain point-gr likelihood
    │  
    │  matcher's peak sits ~200f...
    ├─ hengck23 (2026-06-02 19:30:28.297000) [+1]
    │  Cnn does improve on specific cases and but general cases.
    ├─ hengck23 (2026-06-02 23:54:24.810000) [+0]
    │  examples of different methods
    │  CNN+sdf (using gr) : global gr waveform pattern
    │  
    │  
    │  
    │  transformer on dz (not using gr) : dz prior
    │  
    │  
    │  PF on single value GR :  local  gr match based on local stste (velocit...
├─ wqi876 (2026-06-01 08:22:53.033000) [+2]
│  Thank you very much for your discussion. It has been very helpful to me. And your profile picture is so cute!
├─ hengck23 (2026-06-01 02:29:59.483000) [+2]
│  effects of hacks  
│  
│  no GR features are used.
│  input only use x,y,z,dz,dtvt history, tvt history   
│  
│    
│  
│  left: validation, right: train
│  red: predict, blackL ground truth
│  (do note the scale of the y a...
  ├─ hengck23 (2026-06-01 02:34:15.850000) [+0]
  │  seq = torch.cat([
  │             h_dtvt_history.reshape(B,H,1),
  │             h_tvt_mask.reshape(B,H,1),
  │             h_dz.reshape(B,H,1),
  │             h_x.reshape(B,H,1),
  │             h_y.reshape(B,H,1),
  │       ...
├─ hengck23 (2026-05-25 11:16:58.867000) [+5]
│  i discover a hack!
│  
│  
│  
│  first fig: dz
│  second fig: dtvt
│  why? annotation leak!  (that is how starsteer works)
  ├─ hengck23 (2026-05-25 11:21:05.803000) [+1]
  ├─ Tom (2026-05-25 11:41:34.557000) [+2]
  │  The red/blue = direction segments sign(dtvt). My test just confirmed the structure underneath it: ANCC (formation top) is ~piecewise-linear with ~15 control points per well (~323 rows apart). That ...
    ├─ hengck23 (2026-05-25 12:05:10.733000) [+1]
    │  maybe just prediction dtvt = a(dz)*dz. i.e. your network predict dtvt and use both local dtvt loss and global cumsum tvt loss
    ├─ sleep3r (2026-05-25 12:23:26.710000) [+1]
    │  yeah, this seems real. I tried using ANCC only as a train-time teacher:
    │  
    │  target: sign(dANCC) = down/flat/up
    │  features: test-safe MD/X/Y/Z/GR/TVT_input only
    │  
    │  5-fold OOF hidden direction accuracy is ~...
    ├─ hengck23 (2026-05-25 12:42:52.793000) [+2]
    │  i plot dz and dtvt on the same plot. they are the same scale !!!!  maybe competition will reset
    ├─ hengck23 (2026-05-25 12:50:52.837000) [+3]
    │  h_tvt = h["TVT"].values
    │      h_z = h["Z"].values
    │      h_md = h["MD"].values
    │      h_dtvt = np.gradient(h_tvt)
    │      h_dz   = np.gradient(h_z)
    │  
    │      plt.plot(h_md, -h_dz)
    │      plt.plot(h_md,  h_dtvt)
    │      pl...
    ├─ Tom (2026-05-25 12:58:20.330000) [+2]
    │  −dz and dtvt being the same scale and overlapping in long stretches means: wherever the formation is flat, dtvt = −dz exactly (dANCC=0 → TVT = −Z + C). They only diverge at dip events (your ~15 con...
    ├─ Tom (2026-05-25 12:59:37.323000) [+1]
    │  time to reset now
    ├─ hengck23 (2026-05-25 13:10:10.143000) [+1]
    │  h_dtvt = np.gradient(h_tvt)
    │      h_dz   = np.gradient(h_z)
    │  
    │      H_unknown = len(h_tvt) - h_ps
    │      truth_tvt = h_tvt[h_ps:]
    │      ##---
    │      #find offset
    │      offset = h_dtvt[h_ps-500:]+h_dz[h_ps-500:]
    │   ...
    ├─ hengck23 (2026-05-25 13:24:11.323000) [+1]
    │  i think the offset could be fixed values. my experiments seems to suggest they are limited to set of values
├─ hengck23 (2026-05-26 01:20:45.760000) [+4]
  ├─ Tom (2026-05-26 01:56:35.577000) [+1]
  │  cumsum(−dz − offset) with a discrete offset  =&gt; 7.7 rmse
    ├─ hengck23 (2026-05-26 02:06:20.040000) [+0]
    │  Just need a classifier to choose global offset
    ├─ sleep3r (2026-05-26 08:53:48.643000) [+1]
    │  a fine offset-grid oracle gives ~7.64 RMSE on train hidden rows for me
    │  
    │  but choosing the offset is the hard part: known-prefix offset gives ~37-39 RMSE, and my fold-safe selector only gets ~14.8. S...
    ├─ Tom (2026-05-26 09:01:18.850000) [+0]
    │  Fuzzy inference or mixture desnity network would help
    ├─ hengck23 (2026-05-26 15:24:48.587000) [+0]
    │  The first try should be :
    │  
    │  1) given current location s
    │  2) given a list of offset = -0.1 to 1.0
    │  3) given a list of  future location s1 = 25,50,75, 100, ... 300
    │  4) compute tvt rmse for each candidate...
    ├─ hengck23 (2026-05-26 15:30:35.663000) [+0]
    │  brute force search is  12.18 for one fold
    │  
    │      t = pd.read_csv(f"{KAGGLE_DIR}/train/{sample_id}__typewell.csv")
    │      h = pd.read_csv(f"{KAGGLE_DIR}/train/{sample_id}__horizontal_well.csv")
    │      h_ps ...
    ├─ hengck23 (2026-05-26 15:34:45.773000) [+2]
    │  there is a mxiture/DP transformer that chatgpt recommend:
    │  
    │  Lattice Deduction Transformers
    │  https://arxiv.org/html/2605.08605v1
    │  
    │  class RogiiLatticeTransformer(nn.Module):
    │      """
    │      Simple lattice t...
  ├─ Tucker Arrants (2026-05-29 03:05:08.587000) [+-1]
  │  I think they need to reset. Surely providing the post-PS trajectory (X/Y/Z) is a problem? It's causally downstream of the answer - the driller steered based on where the formation actually was, so ...
    ├─ hengck23 (2026-05-29 03:11:07.863000) [+1]
    │  Not direct nor obvious answer. Still needs some clever hack to work. But does make getting answer easier.
    ├─ PatrickAIForFun (2026-05-29 06:46:09.960000) [+2]
    │  I don't think a reset is necessary. If you look at all training videos and resources by ROGII one can clearly see that there are two types of geosteering which are done in the real world:
    │  
    │  
    │  
    │  live g...
├─ hengck23 (2026-05-25 05:22:30.283000) [+3]
│  i can do some fast match from visual inspection if i segment the direction of the well
│  
│  
│  
│  
│  look for highest and lowest point
│  
│  check neighbourhood values from that point
│  
│  then you can find large seg...
  ├─ Tom (2026-05-25 07:25:45.937000) [+2]
  │  Developing a “Trace Back” mechanism could further improve the score. One possible approach is to build a dictionary (or bag-of-signals) that serves as a strong reference for matching
    ├─ hengck23 (2026-05-25 10:38:02.067000) [+5]
    │  i suddenly have a cheat method.
    │  
    │  1) you are at typewell location s at PS.   
    │  
    │  2) we are not interested in tracinig the well trajetory. rather we are interested in detecting the max and min offset v...
    ├─ sleep3r (2026-05-25 11:11:55.463000) [+1]
    │  i tried a similar direction: instead of trusting one global GR heatmap, i build local GR-event candidates and then use a chunk-level DP policy to stitch/select a smooth path
    │  
    │  early result: this doe...
├─ hengck23 (2026-05-22 23:54:38.270000) [+6]
│  update on cnn+sdf:
│  
│  
│  
│  some backbone and decoder architecture  are better
│  
│  augmention using flip + different stretch improve results
│  
│  time to spend on generator to generate more possible train data:...
├─ hengck23 (2026-05-22 02:24:40.230000) [+4]
│  one challenge of the competition is to find good representation. Here is using cnn + sdf (signed distance function)
  ├─ Tom (2026-05-22 02:30:02.903000) [+2]
  │  SDF seems like a solid option. This also reminds me of the Vesuvius Challenge, might be able to transfer some tricks from there.
    ├─ hengck23 (2026-05-22 04:19:41.070000) [+5]
    │  @tom99763 
    │  
    │  demo inference and training code are up:
    │  https://www.kaggle.com/code/hengck23/cnn-sdf-example
    │  https://www.kaggle.com/datasets/hengck23/hengck23-rogii-cnn-mtp-demo  (training py file)
  ├─ hengck23 (2026-05-22 04:32:09.797000) [+1]
  │  The fact that CNN can detect micro 2d pattern makes me think that the data are probably synthetic or the signal modelling in geology is really good?
    ├─ hengck23 (2026-05-22 04:36:33.553000) [+1]
    │  i am thinking of predicting the geology plane, eg ANCC = tvt -z instead. such planes are more linear and benefit from sdf (natural smoothness and planar regularisation from ground truth!)
    ├─ Tom (2026-05-22 12:53:40.013000) [+1]
    │  Tvt - z can work better than directly predicting tvt.
    ├─ hengck23 (2026-05-22 14:54:09.883000) [+2]
    │  instead of
    │  
    │  mistfit_gr = t_gr-h_gr
    │  
    │  
    │  use
    │  
    │  mistfit_gr = t_gr- interpolate( h_tvt-well_z, h_tvt, h_gr)
    │  
    │  
    │  maybe you can see a linear zero line (matched gr)
    ├─ Tom (2026-05-22 15:22:53.963000) [+2]
    ├─ hengck23 (2026-05-22 15:41:25.333000) [+3]
    │  i tried some toy data
    ├─ hengck23 (2026-05-22 15:58:11.203000) [+2]
    │  So the pf, k-beam, dp, viterbi etc searches are just detecting lines or multi ple lines hypothesis.
    │  
    │  But there is an issue, ancc plane anchoring means the range of tvt is very small if the geologic...
├─ sleep3r (2026-05-23 18:35:23.623000) [+1]
│  Tried a sliding-window CNN that predicts TVD corrections over the base prior, using horizontal GR + typewell correlation. Added synthetic pretraining to teach the correlation - worked great on synt...
  ├─ sleep3r (2026-05-24 09:55:54.517000) [+2]
  │  I rechecked with real train-well panels. The issue seems not only CNN/MTP capacity: the true TVT path often is not a reliable high-score ridge in the GR/typewell heatmap. In our localized/stretch p...
    ├─ hengck23 (2026-05-24 14:09:20.130000) [+1]
    │  how about let GR = concate (gr values, location values). then each GR value is diiferent. correlation is match of values and distance
    ├─ sleep3r (2026-05-24 15:56:22.933000) [+1]
    │  I tested this exact idea: combine GR matching with a test-safe location prior
    │  
    │  Concatenating / combining location with GR absolutely helps remove global false ridges
    │  But the effect seems to come fr...
    ├─ hengck23 (2026-05-24 23:53:06.373000) [+1]
    │  The problem of geosteering is actually "move the wellbore between the target top and bottom geology region." Hence, here the inversion is localised, where is the wellbore within the layers? You can...
    ├─ hengck23 (2026-05-25 00:19:28.800000) [+1]
    │  check 10a1281a.png in the train dataset 
    │  
    │  
    │  
    │  the reference TW GR signal for matching is only "so short". many of the horziontal GR "windows" are not useful at all except for the peaks
├─ hengck23 (2026-05-20 12:39:39.033000) [+4]
│  what you get if you use unet and do "blood vessel" segmentation
│  
│  validtation
│  
│  
│  
│  training
│  
│  
│      def forward(self, typewell, horizontal, hint):
│  
│          #todo raw signal channel
│          B,T = typewel...
  ├─ hengck23 (2026-05-20 12:47:29.917000) [+1]
  │  i am surprised that some results are perfect and it is bidirectional and needs not to be continuous (e.g. match can happen in the middle of image and propagate out)
  ├─ Tom (2026-05-20 14:40:26.917000) [+2]
  │  might can consider problem as iterative image inpainting I think.
    ├─ hengck23 (2026-05-20 15:53:58.387000) [+2]
    │  I am surprised there is no multiple paths. I only use bce loss. Some paths diverted from the truth with high confidence. It means that if we use gr information, we can very similar train labels tha...
  ├─ hsiaosuan (2026-05-29 01:37:32.887000) [+1]
  │  Reminds me of Vesuvius!!
├─ hengck23 (2026-05-17 14:22:01.793000) [+6]
│  example notebook
│  https://www.kaggle.com/code/hengck23/cnn-mtp-example?scriptVersionId=320093395
├─ hengck23 (2026-05-17 08:52:10.707000) [+5]
│  results is good at least for short-term forecast of 8 future interval steps. (each interval uses average of 32 GR values).
│  here are validation resuits. black is truth, orange is probability weighte...
  ├─ hengck23 (2026-05-17 09:37:13.173000) [+5]
  │  try a longer horizon of future steps=16, history=9. as expected, prediction starts to diverge. but good news is that the truth  path is still predicted as a lower score candiate, eg top-6 solution ...
├─ hengck23 (2026-05-18 08:06:52.470000) [+3]
│  Take-home message: mathematical correlation versus machine-learned correlation.
│  
│  So anything that is imperfect can be made perfect by learning. eg, we have our DTW needs to take care of reverse ind...
  ├─ hengck23 (2026-05-18 08:41:32.733000) [+2]
  │  cnn should be very good to capture these micro box patterns (pairs of 2d signal). these are just like 2d tokens. But i need to recreate ROGII segment endpoints annotations.
  ├─ Tom (2026-05-18 09:08:40.227000) [+1]
  │  Thanks, this is very useful info
├─ hengck23 (2026-05-15 14:25:59.273000) [+5]
│  code and lesson (lecture notes)
│  https://github.com/geosteering-no/inversion_school_geosteering/tree/main
├─ hengck23 (2026-05-16 08:33:05.993000) [+4]
│  i make some plots. i think the formulation is not the issue. the issue is that the data is really noisy. It is difficult for human to match if we only see a window segment of vertical and horizonta...
├─ hengck23 (2026-05-18 06:58:50.437000) [+1]
│  learning distance fields
├─ hengck23 (2026-05-16 10:27:37.610000) [+1]
├─ hengck23 (2026-05-16 10:10:03.043000) [+1]
│  another example
├─ hengck23 (2026-05-16 09:56:52.223000) [+1]
├─ hengck23 (2026-06-08 10:14:32.910000) [+0]
│  i think rmse error is some how biased (e.g. increases with length due to error accumulation. post processing your results may help)
├─ hengck23 (2026-06-04 11:04:28.190000) [+0]
│  plot of tvt max - tvt min verus tvt length
  ├─ hengck23 (2026-06-04 11:14:49.710000) [+0]
├─ hengck23 (2026-06-03 13:33:52.007000) [+0]
│  i show the same solution in two different visualisations. 
│  z prior is much stronger than gr prior.
  ├─ hengck23 (2026-06-03 13:39:17.153000) [+0]
  │  eg, it is "easy" to correct this error?
  │  
  │  the initial offset before PS tells a lot about the distance between well z and geo z
├─ hengck23 (2026-06-03 12:17:33.370000) [+0]
│  maybe this is helpful
├─  (2026-05-26 18:01:38.883000) [+0]
├─  (2026-05-15 14:14:38.087000) [+0]
