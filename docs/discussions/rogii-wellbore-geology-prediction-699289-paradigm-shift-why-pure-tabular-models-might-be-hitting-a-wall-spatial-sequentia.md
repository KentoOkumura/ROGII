# Paradigm Shift: Why pure Tabular Models might be hitting a wall (Spatial & Sequential Context)

- archived_at: 2026-06-11T13:50:25Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699289

Topic #699289: Paradigm Shift: Why pure Tabular Models might be hitting a wall (Spatial & Sequential Context)
  Author: Amged Alfaqih
  Posted: 2026-05-13 12:22:48.528000
  Votes: 21  Comments: 4

Hi everyone,

After spending some time diving deep into the data and experimenting with various preprocessing pipelines, I noticed a common trap we might all be falling into: treating this purely as a standard tabular regression problem.

If you are just passing [X, Y, Z, MD, GR] into LightGBM or XGBoost, you will eventually hit a hard ceiling on your CV/LB score. Why? Because the models are missing the physical and spatial realities of the wellbore trajectory.

1. The Sequential Reality (Physics of the Drill)

We aren't just looking at random rows; we are tracing a path. Features like MD (Measured Depth) and Z dictate a trajectory.

Instead of simple lags and rolling windows, has anyone experimented with Particle Filters (PF) or Beam Search? By treating the expected TVT as a moving particle that updates its state based on the Gamma Ray (GR) observations and spatial constraints, we can create incredibly strong baseline predictions to feed into our Gradient Boosting models.

2. The Spatial Reality (Geology is Continuous)

A well doesn't exist in a vacuum; it shares the same geological formation as its neighbors.

Relying solely on X and Y coordinates in a tree model is inefficient. A better approach is using Spatial Imputation (like cKDTree) to find the nearest known wells and calculate the median TVT or formation depth in that specific localized area.

Feeding this "spatial neighborhood consensus" as a feature massively stabilizes the predictions for unseen evaluation rows.

By shifting the focus from "tuning model hyperparameters" to "building physics-aware and spatial-aware features", the performance jump is massive.

Are you guys currently using any sequential tracking (like Particle Filters/Kalman Filters) or mostly relying on heavy rolling/lag statistical features?

Would love to hear your thoughts!

Comments:
├─ hengck23 (2026-05-13 22:32:03.703000) [+11]
│  Check rogii patent on its product startsteer. Its viterbi/ beam search includes using dip equation
│  US20190106974A1 — “Systems and methods for horizontal well geosteering”
│  US20230019126A1 — “Methods...
├─ Durga Kumari (2026-05-16 18:59:08.737000) [+1]
│  Really good point. I think many people are treating this as a pure tabular problem while the data is clearly sequential + spatial. The idea of combining trajectory-aware features with spatial neigh...
├─ faizan (2026-05-14 18:01:55.147000) [+1]
│  Really well-articulated post, and I think you've put your finger on something a lot of people are quietly running into but haven't named this clearly.
│  The point about sequential context is especial...
  ├─ 想去看海 (2026-05-16 09:35:19.390000) [+-1]
  │  你好，我是初入机器学习的小白选手，对于您的“但值得尝试的一件事是距离加权插值（IDW）而不是平坦的中值和mdash;”这一观点十分赞同，但我还有一个小小的疑问，我该怎么寻找附近的井？是根据每个样本的xyz值吗？这样的话是不是要将train中所有的样本数据综合起来分析？
