# ML to learn generator for forward simulation

- archived_at: 2026-06-11T13:48:18Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/702474

Topic #702474: ML to learn generator for forward simulation
  Author: hengck23
  Posted: 2026-05-24 01:56:40.313000
  Votes: 12  Comments: 16

I ask chatgpt to code ML to learn generator for forward simulation (eg for augmentation or verification of solution goodness). He suggested two conditional CVAE, one for trajetory and another for GR. here are the results







not good enough yet but a good initial generator considered that i did not write any code  

the more important thing is to plot the trajectories in latent space for train and validation (and also probe hidden test)

then see if "generated train+train" covers validation (and hidden test)

Comments:
├─ hengck23 (2026-06-11 01:40:07.497000) [+2]
│  physics-based particle generation:
│  
│     #piecewise modeling -------------------------
│      xk = np.array([0, 250, 500, 750, 1000], dtype=float) #5 control points
│      yk = np.random.uniform(0, 20, (30...
├─ hengck23 (2026-06-09 10:59:04.230000) [+1]
│  i make a unet to predict sdf. the training data are actually validation samples with simulated TVT (by samping) and simulated GR (by interp). Then the trained model is tested on actual GR(form csv)...
  ├─ Tom (2026-06-09 11:49:54.297000) [+4]
  │  Share my results:
    ├─ hengck23 (2026-06-09 21:03:27.637000) [+1]
    │  Thanks. this is the power of CNN. global optimization. and the power of training 
    │  
    │  i think you can get better results with dz constraint (i.e. physics modeling). i am still figuring out physics-con...
├─ hengck23 (2026-06-08 20:19:25.213000) [+1]
│  i did an interesting experiment:
│  1) train a model with train set and validation set. i identify an validation example E with high rmse error  (e.g. greater than 30)
│  2) repeat with train=train+E. no...
  ├─ hengck23 (2026-06-08 20:34:27.913000) [+0]
  │  easy snap-in code for experiment:
  │  
  │      def __getitem__(self, idx):
  │          sample_id = self.sample_id[idx] 
  │          if self.cache.get(sample_id) is None:
  │              load_one(sample_id, self.cache)...
├─ hengck23 (2026-06-08 07:21:01.177000) [+1]
│  you actually have both forward and backward physics models!
  ├─ Kuni05 (2026-06-08 23:56:24.083000) [+0]
  │  Hi！ I have read your idea in other discussion, but backward model is little useful in my model (Using the public LB as a reference), very small improvements even the RMSE increased. But in local te...
├─ hengck23 (2026-06-08 06:31:44.047000) [+1]
│  any advice from geological experts on how to generate a synthetic gr log?
├─ hengck23 (2026-06-11 11:12:47.347000) [+0]
│  1d unet diffuser (PDDM)  
│  
│  let observed gr = simulated gr + noise. simulated gr = inter(tvt candates, typewell gr and tvt). noise =  observed gr - simulated gr. Then noise model is noise = DPPM( si...
├─ hengck23 (2026-06-10 01:22:46.003000) [+0]
│  my experiment on physics constrained simulation. Again no training data, just simulate tvt and gr from validation data as training set.
│  
│  at first i though the search space was large, but then we ne...
  ├─ hengck23 (2026-06-10 06:35:03.787000) [+1]
  │  i thought the samples may be confusing one another. so i did another example to simulate only one sample per net.
  │  
  │  
  │  
  │  
  │  
  │  this is probably the limit when no noise modeling is used.
├─ hengck23 (2026-06-09 21:11:29.853000) [+0]
│  using linear to model geological plane: oracle validation (using best-fitted geo plane) 4.8203103565376715 (expected lb  = 4.8203103565376715+2)
│  
│  search range
│  
│    best_rmse =1e100
│      best = None
│    ...
  ├─ Philippe Lonjoux (2026-06-11 09:36:31.280000) [+0]
  │  Thanks for sharing the full thread — it was genuinely insightful to follow the progression from CVAE to piecewise-linear planes, and then to the linear-plane oracle. The point about GR only giving ...
    ├─ hengck23 (2026-06-11 10:05:05.280000) [+0]
    │  output space parameterisation
    │  
    │  "The linear plane oracle gets down to 4.82 ft on validation,"  
    │  
    │  this is easy to model the geological plane but difficult to score (the interpd GR may not match the  ...
    ├─ Philippe Lonjoux (2026-06-11 11:52:45.430000) [+0]
    │  Thanks for the detailed reply — this clarifies a lot, especially the experiment breakdown and the focus on max RMSE as a way to enforce a strong prior.
    │  
    │  Quick follow-up on the noise modelling: have...
