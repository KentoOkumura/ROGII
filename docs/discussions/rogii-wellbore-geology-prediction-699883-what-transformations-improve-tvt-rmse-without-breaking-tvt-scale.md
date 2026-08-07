# What transformations improve TVT RMSE without breaking TVT scale?

- archived_at: 2026-06-11T13:50:27Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699883

Topic #699883: What transformations improve TVT RMSE without breaking TVT scale?
  Author: POR160893
  Posted: 2026-05-15 16:10:57.955000
  Votes: 6  Comments: 2

After ~35–40 phases of analogue/retrieval/residual experiments, I think the main unresolved question is no longer “how do we fit GR better?”, but:

What transformations of a strong TVT anchor actually improve hidden TVT RMSE without destroying absolute TVT scale?

A lot of my earlier experiments implicitly assumed:



better GR match -&gt; better TVT

better local sequence analogue -&gt; better submission


But discussion here (especially around inverse problems / priors) suggests this is not necessarily true.

What I’m seeing empirically:



Strong anchor submissions (Phase23D-style smooth constrained solutions) remain surprisingly robust.

Aggressive free-form transformations can catastrophically fail even when pseudo-validation looks good.

Small flatten/compression variants sometimes move LB slightly, but signal is weak.

Preserving global TVT statistics (mean/std/range/drift) seems extremely important.

Local GR resemblance alone does not appear sufficient.


So I’m trying to understand the correct search space.

For those experimenting successfully:



Are you mostly applying:




affine transforms?

local warping?

slope regularization?

residual learning?

monotonicity priors?

MD-domain smoothing?

sequence transplantation?




How tightly are you constraining:




per-well mean?

std/range?

drift?

curvature/roughness?




Has anyone found evidence that:




hidden RMSE rewards smoother geological priors more than local GR fit?

preserving absolute TVT scale matters more than matching local structure?




Most importantly:
If you start from a “good anchor”, what transformations have actually improved LB consistently rather than randomly?


At this point I suspect the competition is largely about learning the correct prior space for TVT, not maximizing GR similarity.

Comments:
├─ hengck23 (2026-05-15 21:13:47.257000) [+3]
│  code and lesson (lecture notes) https://github.com/geosteering-no/inversion_school_geosteering/tree/main
│  
│  the above answers your questions.
│  
│  1) you need to find a solution without GR first (i sugge...
├─ hengck23 (2026-05-15 19:33:32.320000) [+2]
│  You can still use GR for fitting, but the goal should be to fit the geological structure represented by the GR log, not merely the raw GR values.  
│  
│  For example, the peaks and valleys in a GR log m...
