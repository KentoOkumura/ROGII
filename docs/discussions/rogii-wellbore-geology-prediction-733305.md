# 11th Place Solution

- archived_at: 2026-08-06T13:07:51Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/733305

Topic #733305: 11th Place Solution
  Author: IIDA
  Posted: 2026-08-06 12:19:02.601000
  Votes: 5  Comments: 0

11th Place Solution (solo) — Using the Well's Own Known Section as the Reference

First of all, thank you to ROGII and the Kaggle team for a physical, deep, and suspenseful competition — and congratulations to the winners.

On the public LB, even my best submission scored 6.280 — rank 207. The submission that actually counted sat far below that: 6.462, roughly 1240th-equivalent on public. On private it scored 6.299 and finished 11th.

1. The task and the data

The task is to predict TVT — the depth of the wellbore relative to the geological layers — at every point of the later section of each horizontal well (drilled, but with the answer hidden). The inputs are the well path (X/Y/Z/MD), the GR log, the true TVT given only for the first section, and a reference log measured in a nearby vertical well (the typewell).


Fig. 1: A training well. Left = the typewell (GR vs depth). Top right = the well trajectory (black = known section with TVT given, red dashed = prediction target). Bottom right = GR measured along the well. The task is to infer the depth of the red dashed curve by matching this GR against the typewell.

Before building models, I cross-checked the labels against the GR. Three observations set the whole strategy.

(1) In many wells, GR alone cannot pin down the depth.
For many wells, several interpretations coexist that keep almost the same GR shape but sit a few feet apart in depth. On easy wells you can pick the right one from the GR; on the hard wells it was close to a coin flip. Chasing the exact answer there is hopeless, so I chose to sit between the plausible interpretations and limit the damage when wrong. Since the metric is RMSE (squared error), cutting the size of the big misses matters more than being right more often.

(2) The typewell is less reliable than it looks.
The typewell is logged in a different well, so instrument differences, calibration, and lateral geological change all shift it relative to the lateral GR. Comparing the two at the same depth, they disagreed far more than I expected. Every idea of the form "match against the typewell harder" ran into this wall.

(3) Test wells are more isolated than training wells.
Test wells tend to sit where there are no nearby training wells. Methods that borrow information from neighboring wells work in CV and then do nothing on the test set. I used well-level GroupKFold CV to reproduce this situation as closely as I could.

2. Solution overview

Rather than polishing a single model, I judged that this problem rewards mixing several models that fail in different ways. Because of the depth ambiguity (observation 1), any single model will miss big somewhere; averaging models that miss in different ways softens those misses.


Fig. 2: The pipeline (simplified). The main engine is a UNet solving the well as an image; particle filters and GBDT add predictions of a different nature, and everything is blended per well.

The blend weights are not uniform — they change well by well. Wells where the models agree are averaged straightforwardly; wells where the predictions split lean toward the safe side. Finally, the predicted trajectory is smoothed into a gentle curve: the true layer boundaries are gentle, so fine jitter in the prediction is nothing but error.

The final version scored CV 5.66 (773 wells, GroupKFold).

3. Main model: solve the well as an image with a UNet

This formulation is the same family as Tucker Arrants' 26th-place solution; his forum posts helped me a lot.

Build an image with depth on the vertical axis and position along the well on the horizontal axis. Paint each pixel with a matching score — "if the well were at this depth here, how well would its GR agree with the typewell?" — and the correct trajectory shows up as a dark valley in the image. Feed this to the UNet and have it decide, column by column, which depth is correct. The output is a likelihood for each depth, so its weighted average becomes the prediction. One pass predicts the whole well.


Fig. 3: The matching-score image for a real well (dark = GR matches the typewell well). The truth (red dashed) runs inside a dark band — but several other dark bands run in parallel. This is the depth ambiguity of observation (1): the model often cannot decide which band to ride.

Three things I did on top of that:



Auxiliary inputs: besides the matching-score image, I also feed the raw GR and typewell, hints from the known section, and the trajectory slope.

Averaging several models: I trained models independently with slightly different settings and averaged them. Unglamorous, but the best cost-performance of anything here.

Pretraining on synthetic wells: generate artificial layers, trajectories and GR, pretrain on that, then train on the real data. The generator settings were matched to statistics of the real data.


4. The idea that mattered most: reference the well itself, not the typewell

In this problem, the true TVT of the first section of every well is given. That means you hold a GR-vs-depth table measured in that very well. Unlike the typewell, it is measured in the same well, with the same tool and the same calibration — so the mismatch from observation (2) simply does not exist.


Fig. 4: The prediction-target section of a training well (gray dots = true depth vs GR), overlaid with the typewell (blue) and the self-reference built from the known section (red). Scoring how close each is to the true GR: on this well, typewell 14.4 vs self-reference 6.6 (lower is better). Across the 548 wells I could evaluate, the self-reference was the closer reference in 64% of them.

I used it in two ways. First, I added the matching image against this self-reference as an extra UNet input — the single most effective improvement late in the competition. Second, I swapped the reference of the particle filter (a method that probabilistically tracks depth candidates) from the typewell to this one. The self-reference only exists over the depth range covered by the known section, so outside that range I fall back to the typewell.

No comments
