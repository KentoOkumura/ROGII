# Curious how others are handling outlier wells in the spatial ANCC index

- archived_at: 2026-06-11T13:50:23Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700285

Topic #700285: Curious how others are handling outlier wells in the spatial ANCC index
  Author: Jordan
  Posted: 2026-05-17 04:41:09.386000
  Votes: 2  Comments: 0

I'm using a centroid-level Kriging model (one point per well) to predict ANCC at test/training wells then feeding that into an ANCC-Z baseline. When I look at the spatial residuals, about 5-6% of wells have ANCC values that are significant outliers relative to their 10 nearest neighbors. Has anyone found that keeping or removing outliers has had meaningful differences?

I'm second guessing myself a bit here because if the wells reflect authentic geological observations and aren't data errors, it would likely make the spatial model worse in areas where it matters the most

No comments
