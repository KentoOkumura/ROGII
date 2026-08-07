# 13th Place Solution

- archived_at: 2026-08-06T13:11:54Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/733174

Topic #733174: 13th Place Solution
  Author: monnu
  Posted: 2026-08-06 02:55:57.611000
  Votes: 15  Comments: 0

13th Place Solution

Thanks to the hosts, and to everyone who shared what they were finding along the way.
I competed solo and finished at 6.319 private (13th) / 6.008 public.



Overview



Put every pair of (typewell depth, horizontal position) into a single image, and solve it as a per-column classification problem: for each MD column, pick the correct TVT row.

Pretrain on physically consistent synthetic data, then fine-tune on the real wells.

At inference, don't read the field row by row.
Choose one trajectory from a bank that is coherent over the whole well (coherent decode).

Blend three image legs with one 1D candidate leg from a different lineage.


alignment image 512x768
      |   ConvNeXt-T / EfficientNet-B4 / HRNet-W18 (ImageNet pretrained)
field = 512-class CE per column
      |   coherent decode
blend of 3 image legs + 1D candidate leg


Solution

Input

The typewell is resampled to 0.5 ft per row for 512 rows; the horizontal well becomes 768 columns.
Twelve channels: both GR logs, their difference, a history line over the known section, and the well geometry.
No offset wells, formation tops, Geology, or spatial field.

I did not go with windowed matching because the prediction section spans about 4,900 ft in MD while the TVT range is only 26 ft at the median.
Take a 300 ft window in MD and the reference log moves roughly 1 ft, so the reference is nearly constant inside the window.
Laying out all pairs at once and letting the convolutions resolve the global alignment was the way around it.

Training

Two stages, starting with synthetic pretraining.
At first I was transplanting TVT trajectories directly from other wells.
Keeping the target well's own typewell and geometry, and borrowing only the geological-coordinate trajectory U = TVT + dZ from a donor so that TVT = U - Z is reconstructed, took it from 12.641 to 8.981.

It was also possible to measure that the synthetic data was too easy: the true offset was top-1 for 100% of synthetic wells against 85% of real ones.
So GR gain/offset/drift and missing-data gaps drawn from the real wells went in as augmentation on the real side.

Then fine-tuning on real data.
One well is one sample, and the PS position is redrawn every time.
Continuing to mix synthetic batches in during fine-tuning was still worth +0.228 at the end.

Coherent decode

Reading the field row by row gives a trajectory that is not physically possible for the well as a whole.
Instead I build an analytic trajectory bank (cubic splines, 1,536 of them) with the held fold excluded, keep 127 of them by field likelihood, and normalise once over the whole well before taking the posterior mean.
This alone was 6.219 to 5.859.

Blend

Alongside the three image legs I kept one 1D pipeline of a different lineage, which generates candidate trajectories and has a verifier choose among them.
It is about 1.4 ft weaker on its own, but its errors are the least correlated with the image legs, and it was the most valuable member of the blend.
Weights were fitted fold-out with the held fold excluded.

CV and LB

CV is the pooled RMSE over all held rows from 773 wells, on a fixed GroupKFold(5).





CV
public
private




ConvNeXt alone
6.219
6.364
6.931


+ coherent decode
5.799
6.171
6.697


+ synthetic / augmentation
5.534
6.122
6.181


+ reworked fine-tuning
5.268
5.868
6.451


blend (final submission)
5.097
6.008
6.319




The final submissions were chosen on CV alone; I did not look at the public LB.

Thanks for reading.

No comments
