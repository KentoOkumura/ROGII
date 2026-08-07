# 10th Place Solution: A Compass, Not a Map

- archived_at: 2026-08-06T13:07:51Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/733315

Topic #733315: 10th Place Solution: A Compass, Not a Map
  Author: Can
  Posted: 2026-08-06 12:55:16.244000
  Votes: 1  Comments: 1

10th place, solo: a compass, not a map

I spent ten weeks predicting where a drill bit sat inside the rock. The measurement that decided this
competition for me was not a model. It was working out which of my own numbers was lying.

Result: 10th private, 21st public, 6,191 teams, solo entry.

Everything below is out of fold over 773 wells and 3,783,989 scored rows. I rebuilt every number from
the raw arrays before I believed it. The plotting script that made all the figures is at the end.



1. Two facts that shaped everything

TVT + Z is a geological surface

Take TVT, add Z, then subtract the formation top depth that the training set gives you. The result is
constant inside a well to 0.07 ft, which is the rounding in the CSV. This holds for all 766 wells that
have that column.

So the real object is:

$$A(MD) = TVT(MD) + Z(MD)$$

This is the height of a rock surface, sampled along the well path. Predicting TVT means extrapolating a
smooth 3-D surface, then subtracting the Z you were already given.

The well steers, so the target barely moves



Over the scored part of a typical well:




quantity
median swing
p95




Z, the wellbore
160.6 ft
270.5


A, the rock surface
157.5 ft
266.1


TVT, the answer
26.4 ft
54.4






The crew is doing the work. They steer all the time to stay inside a layer that is moving. What you have
to predict is the small difference between two large motions that almost cancel out.

Why this matters for design. If you build a state space on A, you spend all your resolution on the
160 ft of geometry that you were given for free. If you build it on TVT, you spend it on the 26 ft you
actually have to find. The smoothness prior still belongs on dA/dMD, the geological dip. I got this
backwards twice and it cost me two working decoders.

Numbers for anyone building a lattice: dA/dMD p1 / median / p99 = -0.065 / +0.017 / +0.072. Standard
deviation of dTVT per 20 ft node = 0.468 ft. Standard deviation of the dip change per node = 0.0096.



2. Where the error lives

It grows like a random walk from the heel






distance past the heel
first 50 ft
100 ft
500 ft
1,500 ft
3,000 ft
5,000 ft




mean |error| (ft)
0.32
0.60
2.31
3.18
4.20
4.55




The signed mean error is close to zero at every distance (-0.03 to -0.26). There is no bias to remove.
There is only build up. After I remove a per well line, the residual autocorrelation is 0.9999 at lag 1,
0.8687 at lag 100, and only 0.1345 at lag 500. So the walk has a correlation length of about 400 to
500 ft.

89% of all squared error sits beyond 1,500 ft from the heel.

And it sits in very few wells






slice
share of total squared error




worst 10 wells (1.3% of rows)
22.7%


worst 39 wells (5%)
43.9%


worst 100 wells
62.4%




Per well RMSE at p5 / p25 / p50 / p75 / p95 = 1.22 / 2.14 / 3.31 / 5.62 / 11.08. If those 39 wells had
only matched the median well, the pooled score would drop from 5.70 to about 4.39. That is larger than
every modelling gain of the whole competition put together, several times over.

What each shape of correction could reach



These oracles are allowed to look at the answer, so they tell you what a parameterisation is worth before
you build it:




correction
pooled




flat continuation
15.91


per well constant
9.04


per well line
6.70


per well quadratic
5.34


per well cubic
4.42




On A with piecewise linear fits: one anchored line 7.59, one free line 5.96, two segments 2.43,
three segments 1.44.

My final 5.72 already beats the best possible single straight line. So anything below about 5.0 has to be
fitting bends, not a drift. This killed the whole "just estimate the dip better" family before I wrote
any of it.



3. The solution, and what each stage was worth

The solution is a classical multi candidate engine with a learned selector, followed by two neural
correctors.

Phase A, physics engine. All 773 training wells' formation top columns are subsampled into
(X, Y, Z_surface) point banks with KD trees, plus a regional polynomial fit. A local plane and IDW
interpolation gives a structural prior for A along the query well. Candidate curves come from a particle
filter with 16 seeds and an HMM forward backward pass over a stratigraphic position lattice, under both
Gaussian and Cauchy likelihoods, with several datum mode variants.

Phase B, 2-D ranking U-Nets. Two families of 5 fold U-Nets read an image built from the lateral GR and
the reference log, and produce a per row distribution over stratigraphic position. Fold logits are summed,
then softmaxed.

Phase C, learned selector. A per well LightGBM scores every candidate curve using about 49 cross
evidence features (agreement between engine variants, NN confidence, GR misfit statistics, lattice
features). It turns the scores into weights with a tempered softmax and blends. A projection then pulls
the blend back toward the structural prior.

Phase C2 and C3, neural correctors. A 23 channel per row tensor feeds a dilated TCN (20 checkpoints
averaged), applied as pred = blend + eta * clip(f, +-20) with eta = 0.10. Then a second U-Net is mixed
in.




stage
pooled CV
change
public




C1, engine + selector + projection
5.8120

5.729


+ TCN row corrector
5.7885
-0.024



+ D2 network, scalar out of fold alpha
5.7544
-0.034
5.648


+ per row disagreement gated alpha
5.7244
-0.030



+ heel anchored gain, s = 1.024
5.7189
-0.006
5.620




The last two rows are the ones I found in the final days. They are the only two gains I could show
transferring: CV improved by 0.0355 and the public score improved by 0.028, on a different set of wells.

The gated alpha, because it is cheap and general. Instead of one blend weight for the second model,
use a per row weight that depends on how much the two models disagree:

alpha_i = clip(b0 + b1 * log1p|d2_i - chain_i|, 0, 0.6)


I fitted (b0, b1) by weighted least squares, out of fold. The values were stable across folds
(b0 from 0.35 to 0.50, b1 from -0.08 to -0.14). Trust the second model where it agrees with the first, and
back off where it does not.

A warning about reading these numbers. Per fold CV of the 5.7544 configuration was
5.50 / 5.99 / 5.65 / 5.51 / 6.09. That is a standard deviation of 0.28, about five times the whole
ladder. Every step above is smaller than the noise in any single fold. Only the paired comparison across
all 773 wells is readable.



4. The decision that actually decided it



Nearby wells share a geological surface. If a well 400 ft away has been interpreted all the way to its
toe, its formation tops should constrain mine. I built that: a stage that anchors the chain to a plane
fitted locally through nearby wells' interpreted tops.

It was the biggest single gain of the whole project. CV went from 5.7544 to 5.5762, a jump of 0.18, at
a time when everything else was worth 0.03. Six independent rebuilds confirmed the arithmetic. A
perturbation test confirmed it was not leaking the query well's own labels: destroying that well's own TVT
and formation tops over the scored zone changed the candidate by exactly 0.000000 ft. Removing duplicate
wells left the gain in place.

Then I submitted it. Public went from 5.648 to 5.977.




configuration
CV
public




no neighbour information
5.7544
5.648


neighbour stage, beta = .075
5.6985
5.683


neighbour stage, beta = .156
5.5762
5.977




Everything in me wanted to call this noise. It was not. On the correct ruler, where both configurations
are scored on the same 52 public wells so the level cancels out, the damage is 2.95 standard
deviations, with an empirical p value of 0.002. All three high weight neighbour submissions land between
2.6 and 3.0 sigma. Their joint probability under noise is 0.0016. The three submissions without the
neighbour stage all calibrate to within 0.2 sigma.

If I solve for the weight that the public score would have preferred, I get beta = -0.032. Negative.

The mechanism turned out to be measurable. The gain sits in wells that have a donor trace close by, and
the hidden set is more isolated than the training set. Median distance to the nearest training well is
683 ft for the held out geometry against 470 ft among training wells themselves. Pricing the stage
under realistic donor density already removed a third of the gain (5.5762 to 5.6291).


  
Cross validation was not wrong. It was answering a question about the training geometry, honestly and
  precisely. I was reading the answer as if it were about somewhere else.


I dropped the whole family. That decision, and not any model, is why this says 10th.



5. The instrument: excellent locally, silent globally

The GR trace should tell you where you are. You match what the tool reads now against the vertical
reference log, and the depth that matches is your position. I built that matcher and spent days sure it
was working, because on ground where I could check it, it was.

Test 1, rigid localisation. Take a well's known section, where the true path is given, and ask the
matcher to locate a rigid shift of that path. 81% of wells land within 1 ft. The median error is
0.25 ft.

Test 2, matched perturbation. Perturb the true path with smooth, heel anchored deformations
(correlation lengths 400, 800 and 1,600 ft, at 2, 5 and 10 ft of RMS) and ask the objective to rank the
truth against them:




correlation length
amplitude
AUC, all rows
AUC, beyond 1,600 ft




400 ft
2 ft
0.884
0.857


400 ft
10 ft
0.947
0.903


800 ft
5 ft
0.911
0.873


1,600 ft
2 ft
0.871
0.847




By every local test I could build, the compass works. And it is worth almost nothing.

Test 3, retrieval. Around my chain prediction, I drew 1,025 smooth heel anchored candidates from an
empirical, fold clean prior. The candidates are real residual shapes taken from wells in other folds and
projected onto a 9 knot spline:





pooled




the chain itself
6.01


best candidate in the set (oracle)
2.60


the candidate the objective picks
7.09


posterior weighted average
6.54




The answer is in the set. The objective cannot find it.

Telling two things apart is not the same as finding one thing among many. Even at 90% pairwise
accuracy, the minimum over a thousand correlated candidates is decided by whichever one got lucky.

And the two skills do not predict each other. How sharply the compass located the truth in a well's known
section predicts that well's final error with a rank correlation of 0.081.


  
I had a compass. The job needed a map.




6. Fourteen directions, each closed with a number




direction
result




Extrapolate the drift from the known section
12.60 (flat = 15.91)


First fifth of the scored zone predicts the last
r = +0.131, slope r = +0.048


Borrow a neighbour's error
r = +0.025 within 2,000 ft


Per well offset from GR matching
r = -0.05, sign agreement at chance


Per well blend weight from 9 observables
5.769 against 5.744 for a fixed scalar


Second order decoder over (TVT, dip)
13.53 (flat = 13.34)


2-D CNN over the GR cost volume
14.5 (predicting zero = 15.80)


HMM forward backward, 9 parameterisations
11.77 (flat = 11.81)


Whole well residual matcher, U-Net + BiGRU
4.952 (predicting zero = 4.918)


Spline coefficients from the cost tube
r about 0 at every knot


Spline coefficients from trajectory geometry
r = -0.02 on average


Six fresh TCN seeds, ensembled
every mixture was worse


Detect the bad wells, then treat them
every treatment was worse


Neighbour stage limited to dense donor wells
gain scales with exposure, not density




Two of these deserve a sentence, because I expected both to work.

Seed ensembling

A top five competitor reported gaining 0.4 from equal weight ensembling. If seeds are exchangeable, then
MSE_K = C + (A - C) / K, and a ten seed average needs a pairwise prediction RMS of about 2.34 ft to
give you that. It needs about 0.71 ft to give you even 0.02. My six independently trained variants
differ by 0.186 ft. They were four times too similar for the smallest useful gain. His single model
had a large removable random component. Mine did not. Measure your own diversity before you copy anyone's
ensembling result.

Triage

Since 5% of wells carry 44% of the error, finding them before scoring should beat any model. The best leak
free detector I built combines engine disagreement with the entropy of the matching cost. It reaches a
rank correlation of 0.41 and catches 4 to 6 of the worst 39, where random would catch 2.

Then every treatment I applied to the flagged wells made things worse. Pulling an uncertain well back
toward its anchor costs more than it saves, because the drift you are refusing is mostly real (a flat
prediction scores 15.91 against my 5.70).

A hard well is not a kind of well. It is an outcome that builds up, and it is not written anywhere in
the well's observable properties at the moment you would need to read it.



7. The shake up, and why I ignored the public board

The public board scores 52 wells. The private board scores 148 different wells. I measured the paired
standard deviation of a difference between two configurations at 0.168 on 52 wells and 0.093 on 148.
So a public gap below about 0.10 carries no information. I wrote that rule down before every submission
and never let the public score break a tie.

The final board was harsher than I expected. The team that led the public board for weeks did not finish
in the top ten. I moved from 21st to 10th without changing my judgement in the last days, only by spending
them removing what a 52 well ruler had rewarded.



8. Acknowledgements

Working alone means the discussion forum was my only window into what the top of the board was doing.
Four people there changed how I spent my time, and it is worth being specific about how.

@tuckerarrants wrote publicly that a single non tabular model can reach a CV below 5, using only the
well and its own reference log, with no neighbour well data at all. That comment did two things for me.
Early on it pushed me toward neural sequence models while I was still adding tabular features. Much later,
when my neighbour based stage improved cross validation by 0.18 and destroyed the public score, his remark
was one of the reasons I believed the public score instead of the CV. If the top of the board did not need
neighbour information, then my stage was probably reading the training geometry rather than the geology.

@ravaghi published the particle filter ensemble that the engine's filter and beam leg is a port of. It
was my starting point for candidate generation and it stayed in the pipeline to the end.

Shrey Gandhi described his segment beam search in enough detail to reproduce, including the ablations
for the persistence prior, the reference log recalibration and the drilled trend centering. It was the
clearest specification anyone posted, and it shaped how I built my own decoder even though mine did not
end up working.

Georgy Mamarin measured something I would not have measured myself: the spread of the public score
across byte identical resubmissions, which is about 0.03. I used that number as a hard threshold for the
whole competition. Any public difference below about 0.10 was treated as unreadable and never allowed to
break a tie. That single measurement is a large part of why I survived the shake up.



9. Coda: the right decision and the better outcome are not the same thing

One more number, because leaving it out would make this a worse document than the work deserves.

The neighbour stage I dropped, the one that gained 0.18 on CV and lost 0.33 on public, the one I killed at
2.95 sigma and never looked at again, scored 6.179 on the private set. My final selection scored
6.269. Shipping the thing I spent a week proving I should not ship would have finished 9th.

I would make the same call again. The evidence at decision time said the stage did not transfer, and it
said so at three sigma across three separate submissions. A process that overrides that to chase 0.18 will
be wrong far more often than it is right. But I am not going to dress up the outcome. Under uncertainty
the correct decision and the better result come apart, and the honest thing is to report both and keep the
process.


  
What I learned is not a model. It is that my cross validation and the hidden set were answering two
  different questions, and that most of the skill here was noticing which one I was being asked.

Comments:
├─ Can (2026-08-06 13:05:19.460000) [+0]
│  One of the biggest reasons I earned my Gold medal today was the incredible insights and educational advice that @tuckerarrants shared throughout the competition. I'd also like to take this opportun...
