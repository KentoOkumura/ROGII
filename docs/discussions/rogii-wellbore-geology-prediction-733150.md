# 9th Place Solution

- archived_at: 2026-08-06T13:09:18Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/733150

Topic #733150: 9th Place Solution
  Author: Anil Ozturk
  Posted: 2026-08-06 01:27:15.721000
  Votes: 26  Comments: 0

First of all, thanks to ROGII and Kaggle for a genuinely unusual competition. And 6K competitors? Whoa. Also congrats to my teammate @datafan07 for the Competition GM title!

Solution TL;DR: Treat the well as an image over (candidate TVT × measured depth), let CNNs assign a categorical TVT distribution per MD column, average everything over 3×5 seeded folds, and blend with position-dependent weights. Trust honest CV over the public leaderboard.



The topic is incomplete, we will be adding details.



1. Autoregressive Approach

This model treats tracking as repeated local classification over candidate offsets.

Input image: For each chunk of the lateral, we build a cost-volume image of shape [9, 64 + 120, 41]: 64 look-back context rows plus a 120-row chunk (MD pooled ×3), against 41 candidate offset lanes spanning ±30 ft around the current anchor. The 9 channels encode GR-vs-typewell match costs and geometry for each candidate lane.

Architecture: 8 dilated Conv2d blocks (64 channels, GroupNorm + GELU, dropout 0.05) followed by a 2-layer BiGRU over the MD axis. Softmax over the 41 lanes, they resemble expected offset per row.



At inference the model runs autoregressively: predict a chunk, re-anchor on its own output, predict the next chunk.

Training samples random (well, start position, anchor noise, prior walk) chunks so the training distribution matches the inference rollout distribution, with a soft-CE + Huber loss.

At inference we average 6 rollout configurations (chunk length ∈ {100, 120, 140} × phase ∈ {0, 40}).


Member OOF: 7.38 ft.
Not our strongest member, but persistently non-zero in every blend fit.



2. Whole-well U-Nets

The strongest members reframe the whole problem as dense segmentation:



Each well becomes a single image. The vertical axis is candidate TVT relative to the anchor (384 bins × 0.75 ft ≈ ±144 ft). The horizontal axis contains the final 128 known MD rows followed by the complete unknown suffix, compressed ×16 into at most 640 columns. 

The network assigns one categorical TVT distribution to every MD column.


Architecture: A ResNet-34-encoder U-Net, column-wise softmax over the 384 depth bins. The nine input channels are typewell GR, lateral GR, signed and absolute GR mismatch, four trajectory-tangent features, and a signed-distance representation of the known-TVT history. Decoding takes the per-column expected row, anchored at the last known TVT.

We introduced two variants for diversity:



well-unet: trained with vertical-shift jitter and horizontal-flip augmentation on the stock image.

relief-unet: same architecture, but with two geometry-warping augmentations applied per sample:


path-relief scaling: Rescale the relief of the target path, re-render the image around the new path with the residual preserved

thickness-warp: A smooth monotone warp of the typewell depth axis, simulating thickness misfit between typewell and well location. The cost channels are recomputed after each warp, so the network trains on physically-consistent perturbed wells rather than pixel noise.


The two U-Nets ended almost identical in RMSE but decorrelated.

2.1 Entropy-weighted TTA

Inference uses 10 views per well: 5 vertical shifts (0, ±12, ±24 bins) × horizontal flip. Stock pooling sums the 10 posteriors equally.

We replaced it with:



Compute each view's mean column entropy, then weight views by softmax(-T × entropy). Here, T acts as an inverse temperature: larger values concentrate more weight on lower-entropy views.

Temperature was tuned per model on OOF (T = 10 for wellunet, T = 5 for reliefunet).





Member
stock TTA
entropy TTA




wellunet OOF
6.33
5.86


reliefunet OOF
~6.0
5.86




2.2 Physics-guided Models

A useful identity reframed the target. Let s(X, Y) denote the mean formation-surface elevation. Within one well:

TVT = s(X, Y) - Z + c_well

The unknown well constant cancels at the last known point, PS:

TVT_i - TVT_PS = (s_i - s_PS) - (Z_i - Z_PS)

Because Z_i - Z_PS is observed, the real unknown is formation-surface drift along the lateral.

Level-slope HMM (drift)

The first deterministic model tracks a joint state of stratigraphic level and local formation-surface slope.

Its transitions integrate the proposed surface slope while incorporating the observed change in Z exactly. Slope changes are kept rare, matching the roughly 340-ft geological segment length observed in the labels.

GR emissions combine two signals:



a weak absolute GR-level match;

a stronger derivative-pattern match, which is less sensitive to GR gain and baseline differences.


The public heel and typewell provide independent GR references. A weak spatial prior is also constructed from train-fold formation surfaces using chord transfer and local weighted least-squares gradient estimates. Forward-backward inference produces a posterior distribution over TVT, and its mean is used as the prediction.

Segmented Beam Interpreter (seg)

The second approach searches directly over piecewise-linear geological interpretations.

Each segment proposes:



a duration of 96, 192, or 384 MD rows;

a formation-surface rate over approximately +/-0.08 ft/ft;

optional jumps between GR correlation cycles.


Candidate paths are scored by matching lateral GR against a composite GR-vs-TVT reference. Where the public heel has swept the relevant TVT range, its same-tool GR provides a sharper reference; elsewhere the typewell supplies coverage.

A beam retains the best 24 partial interpretations. Multiple reference and regularization settings produce nine complete paths. A coarse spatial formation-surface estimate then helps distinguish between GR aliases, followed by a small structural blend. The method is deterministic and requires no neural training.

Posterior Fusion

We also fused models before reducing them to point predictions:

p_fused(y) is proportional to p_HMM(y) * p_other(y)

This produced the fused member from HMM x segmented beam and upoe from HMM x U-Net. When two models agree on a TVT mode, their product concentrates probability there. When they disagree, multiple possibilities remain instead of forcing an early hard choice.

A 30% uniform component prevents either model from completely eliminating a valid mode. This posterior-level fusion was especially useful near the heel and middle of the lateral, while the relief U-Net became the strongest toe specialist.

3. Seed-structured CV and OOF construction

The whole artifact is seed-structured:



3 fold seeds × 5 GroupKFold folds

Test prediction = mean over all 15 checkpoints.


Averaging fold seeds was worth roughly −0.15 pooled RMSE over any single seed.



4. Stack Layer: Banded NNLS + Spread Calibration

Member reliability seemed position-dependent: Trackers were strong near the anchor, whole-well models are strongest mid-lateral and at the toe. So we have tried a banded combiner:



Splits rows into 7 quantile bands of md_since_anchor. Per band, fit a no-intercept, non-negative least squares of the target delta on the raw member deltas.

Computes each row's member std across member deltas, split into 80 quantile bins, and fit a per-bin multiplicative correction m = ⟨p, y⟩ / ⟨p, p⟩ on OOF. It aims to shrink the blend where members disagree.


Each band picks a different specialist model:




band (heel→toe)
segnet
u-net
relief-unet
matcher-family




0 (heel)
0.20
0.09
0.02
high


3 (mid)
0.07
0.26
0.02
med


6 (toe)
0.14
0.00
0.64
low




The relief-trained U-Net owning the toe is consistent with its purpose: path-relief augmentation was designed to prevent the model from under-following large formation movement, whose errors accumulate toward the toe. 👀



5. Validation

We tracked three numbers for every candidate:



chain: in-sample pooled OOF RMSE of the fitted combiner.

heldfold: refit the entire combiner on 4/5 folds, score the held fold, pool.

strict per-foldseed: heldfold where fold seed s uses only fold-seed-s models and boundaries


Final candidates (pooled RMSE): 




Combiner
chain
heldfold
strict (mean over 3 fold seeds)
Public LB
Private LB




plain 4-member NNLS
5.42
5.47
5.56
5.129
6.503


band7+cal80, 8 members
5.06
5.27
5.28
5.435
6.251






Thanks for reading, and congratulations to the winners!

No comments
