# PF baseline got LB 8.863 - any ideas to make it learnable with NN?

- archived_at: 2026-06-11T13:48:23Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/707613

Topic #707613: PF baseline got LB 8.863 - any ideas to make it learnable with NN?
  Author: NobelK
  Posted: 2026-06-11 03:15:20.326000
  Votes: 3  Comments: 6

Hi everyone,

Based on the idea shared in this discussion:
https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700424#3466070

I implemented a baseline using a Particle Filter (PF), and it achieved a public LB score of 8.863.

However, my current approach is still mostly rule-based / inference-only. It does not really learn from the training data. The PF works surprisingly well because it can sequentially track a plausible TVT trajectory using GR likelihood and physical constraints, but I am not sure how to move from this handcrafted PF approach to a learnable model.

My rough idea is to replace the PF with a Neural Network (NN), or at least make some part of the PF pipeline learnable. For example:



train an NN to directly predict the TVT trajectory

train an NN to predict residual corrections on top of the PF output

train an NN to rank/select PF candidate trajectories

learn the transition / observation model used inside the PF

use something like a differentiable particle filter


But I am not sure which formulation is the most practical for this competition. I tried asking LLMs for implementation ideas, but the answers were honestly too generic and not very actionable.

Does anyone have a smart idea, hint, or direction for turning this PF-style baseline into a trainable NN-based approach?

Also, if there is something important I am overlooking in this problem setup, I would really appreciate any comments.

Thanks!

Comments:
├─ Tom (2026-06-11 06:36:19.363000) [+3]
│  There are a lot of options you can do:
│  
│  
│  
│  sub sampling N particles, expand a window then train a network to do N-way classification. This way switching the problem from regression to block-wise inf...
├─ hengck23 (2026-06-11 04:10:39.210000) [+3]
│  first you need to "measure" the performance of your current PF method:
│  1) probability of generating  the "truth trajectory tvt" (or close to it)
│  e.g. you always get at least  30  "truth trajectory ...
  ├─ hengck23 (2026-06-11 04:20:44.013000) [+1]
  │  " I tried asking LLMs for implementation ideas, but the answers were honestly too generic and not very actionable."
  │  
  │  
  │  
  │  hello chatgpt, here is my current PF performance:
  │  
  │  
  │  
  │  probability of generating...
    ├─ NobelK (2026-06-11 04:33:56.073000) [+1]
    │  Thank you so much for the very specific and helpful advice.
    │  
    │  I will re-examine it. Thank you so much.
├─ SpeedSci (2026-06-11 07:33:17.437000) [+-1]
│  Will there be a good single-mode NN open-sourced?
  ├─ SpeedSci (2026-06-11 07:34:30.347000) [+-1]
  │  Find another teammate with tree-based models, and the score will skyrocket.
